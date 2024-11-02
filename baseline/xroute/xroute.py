import copy
import importlib
import json
import math
import pathlib
import pickle
import sys
import time

import nevergrad
import numpy
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

import diagnose_model
import models
import replay_buffer
import self_route
import shared_storage
import trainer


class XRoute:
    """
    Main class to route a chip.

    Args:
        route_module_name (str): Name of the route module, it should match the name of a .py file.

        config (dict, RouteConfig, optional): Override the default routing config.

        split_resources_in (int, optional): Split the GPU usage when using concurent xroute instances.

    Example:
        >>> xroute = XRoute("net_order")
        >>> xroute.train()
        >>> xroute.test(render=True)
    """

    def __init__(self, route_module_name, config=None, split_resources_in=1):
        # Load the routing environment and the config from the module of "route"
        try:
            route_module = importlib.import_module(route_module_name)
            self.Route = route_module.Route
            self.config = route_module.RouteConfig()
        except ModuleNotFoundError as err:
            print(
                f'{route_module_name} is not a supported route module name, try "net_order" or refer to the documentation for adding a new route.'
            )
            raise err

        # Overwrite the config
        if config:
            if type(config) is dict:
                for param, value in config.items():
                    if hasattr(self.config, param):
                        setattr(self.config, param, value)
                    else:
                        raise AttributeError(
                            f"{route_module_name} config has no attribute '{param}'. Check the config file for the complete list of parameters."
                        )
            else:
                self.config = config

        # Fix random generator seed
        # numpy.random.seed(self.config.seed)
        # torch.manual_seed(self.config.seed)

        seed = self.config.seed
        # random.seed(seed)
        # os.environ['PYTHONHASHSEED'] = str(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Remove randomness (may be slower on Tesla GPUs)
        # https://pytorch.org/docs/stable/notes/randomness.html
        # if seed == 0:
        #     torch.backends.cudnn.deterministic = True
        #     torch.backends.cudnn.benchmark = False

        # Manage GPUs
        if self.config.max_num_gpus == 0 and (
            self.config.selfroute_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            raise ValueError(
                "Inconsistent RouteConfig: max_num_gpus = 0 but GPU requested by selfroute_on_gpu or train_on_gpu or reanalyse_on_gpu."
            )
        if (
            self.config.selfroute_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            total_gpus = (
                self.config.max_num_gpus
                if self.config.max_num_gpus is not None
                else torch.cuda.device_count()
            )
        else:
            total_gpus = 0
        self.num_gpus = total_gpus / split_resources_in
        if 1 < self.num_gpus:
            self.num_gpus = math.floor(self.num_gpus)

        if self.config.ray_debug:
            # Debug mode
            ray.init(num_gpus=total_gpus, ignore_reinit_error=True, local_mode=True)
        else:
            ray.init(num_gpus=total_gpus, ignore_reinit_error=True)

        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = {
            "weights": None,
            "optimizer_state": None,
            "total_reward": 0,
            "episode_length": 0,
            "mean_value": 0,
            "training_step": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "num_routes": 0,
            "num_routed_steps": 0,
            "num_reanalysed_routes": 0,
            "terminate": False,
        }
        self.replay_buffer = {}

        cpu_actor = CPUActor.remote()
        cpu_weights = cpu_actor.get_initial_weights.remote(self.config)
        self.checkpoint["weights"], self.summary = copy.deepcopy(ray.get(cpu_weights))

        # Workers
        self.self_route_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def train(self, log_in_tensorboard=True):
        """
        Spawn ray workers and launch the training.

        Args:
            log_in_tensorboard (bool): Start a testing worker and log its performance in TensorBoard.
        """
        if log_in_tensorboard or self.config.save_model:
            self.config.results_path.mkdir(parents=True, exist_ok=True)

        # Manage GPUs
        if 0 < self.num_gpus:
            num_gpus_per_worker = self.num_gpus / (
                self.config.train_on_gpu
                + self.config.num_workers * self.config.selfroute_on_gpu
                + log_in_tensorboard * self.config.selfroute_on_gpu
                + self.config.use_last_model_value * self.config.reanalyse_on_gpu
            )
            if 1 < num_gpus_per_worker:
                num_gpus_per_worker = math.floor(num_gpus_per_worker)
        else:
            num_gpus_per_worker = 0

        # Initialize workers
        self.training_worker = trainer.Trainer.options(
            num_cpus=0,
            num_gpus=num_gpus_per_worker if self.config.train_on_gpu else 0,
        ).remote(self.checkpoint, self.config)

        self.shared_storage_worker = shared_storage.SharedStorage.remote(
            self.checkpoint,
            self.config,
        )
        self.shared_storage_worker.set_info.remote("terminate", False)

        self.replay_buffer_worker = replay_buffer.ReplayBuffer.remote(
            self.checkpoint, self.replay_buffer, self.config
        )

        if self.config.use_last_model_value:
            self.reanalyse_worker = replay_buffer.Reanalyse.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.reanalyse_on_gpu else 0,
            ).remote(self.checkpoint, self.config)

        self.self_route_workers = [
            self_route.SelfRoute.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.selfroute_on_gpu else 0,
            ).remote(
                self.checkpoint,
                self.Route,
                self.config,
                self.config.seed,
                # self.config.seed + seed,
                seed,
            )
            for seed in range(self.config.num_workers)
        ]

        # Launch workers
        [
            self_route_worker.continuous_self_route.remote(
                self.shared_storage_worker, self.replay_buffer_worker
            )
            for self_route_worker in self.self_route_workers
        ]
        self.training_worker.continuous_update_weights.remote(
            self.replay_buffer_worker, self.shared_storage_worker
        )
        if self.config.use_last_model_value:
            self.reanalyse_worker.reanalyse.remote(
                self.replay_buffer_worker, self.shared_storage_worker
            )

        if log_in_tensorboard:
            self.logging_loop(
                num_gpus_per_worker if self.config.selfroute_on_gpu else 0,
            )

    def logging_loop(self, num_gpus):
        """
        Keep track of the training performance.
        """
        # Launch the test worker to get performance metrics
        if not self.config.one_env_instance_only:
            self.test_worker = self_route.SelfRoute.options(
                num_cpus=0,
                num_gpus=num_gpus,
            ).remote(
                self.checkpoint,
                self.Route,
                self.config,
                self.config.seed,
                # self.config.seed + self.config.num_workers,
            )
            self.test_worker.continuous_self_route.remote(
                self.shared_storage_worker, None, True
            )

        # Write everything in TensorBoard
        writer = SummaryWriter(self.config.results_path)

        print(
            "\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n"
        )

        # Save hyperparameters to TensorBoard
        hp_table = [
            f"| {key} | {value} |" for key, value in self.config.__dict__.items()
        ]
        writer.add_text(
            "Hyperparameters",
            "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
        )
        # Save model representation
        writer.add_text(
            "Model summary",
            self.summary,
        )
        # Loop for updating the training performance
        counter = 0
        keys = [
            "total_reward",
            "episode_length",
            "mean_value",
            "training_step",
            "lr",
            "total_loss",
            "value_loss",
            "reward_loss",
            "policy_loss",
            "num_routes",
            "num_routed_steps",
            "num_reanalysed_routes",
        ]
        info = ray.get(self.shared_storage_worker.get_info.remote(keys))
        try:
            while info["training_step"] < self.config.training_steps:
                info = ray.get(self.shared_storage_worker.get_info.remote(keys))
                writer.add_scalar(
                    "1.Total_reward/1.Total_reward",
                    info["total_reward"],
                    counter,
                )
                writer.add_scalar(
                    "1.Total_reward/2.Mean_value",
                    info["mean_value"],
                    counter,
                )
                writer.add_scalar(
                    "1.Total_reward/3.Episode_length",
                    info["episode_length"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/1.Self_routes",
                    info["num_routes"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/2.Training_steps", info["training_step"], counter
                )
                writer.add_scalar(
                    "2.Workers/3.Self_routed_steps", info["num_routed_steps"], counter
                )
                writer.add_scalar(
                    "2.Workers/4.Reanalysed_routes",
                    info["num_reanalysed_routes"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/5.Training_steps_per_self_routed_step_ratio",
                    info["training_step"] / max(1, info["num_routed_steps"]),
                    counter,
                )
                writer.add_scalar("2.Workers/6.Learning_rate", info["lr"], counter)
                writer.add_scalar(
                    "3.Loss/1.Total_weighted_loss", info["total_loss"], counter
                )
                writer.add_scalar("3.Loss/Value_loss", info["value_loss"], counter)
                writer.add_scalar("3.Loss/Reward_loss", info["reward_loss"], counter)
                writer.add_scalar("3.Loss/Policy_loss", info["policy_loss"], counter)
                print(
                    f'Last test reward: {info["total_reward"]:.2f}. Training step: {info["training_step"]}/{self.config.training_steps}. Routes: {info["num_routes"]}. Loss: {info["total_loss"]:.2f}',
                    end="\r",
                )
                counter += 1
                time.sleep(60)
        except KeyboardInterrupt:
            pass

        self.terminate_workers()

        if self.config.save_model:
            # Persist replay buffer to disk
            path = self.config.results_path / "replay_buffer.pkl"
            print(f"\n\nPersisting replay buffer routes to disk at {path}")
            pickle.dump(
                {
                    "buffer": self.replay_buffer,
                    "num_routes": self.checkpoint["num_routes"],
                    "num_routed_steps": self.checkpoint["num_routed_steps"],
                    "num_reanalysed_routes": self.checkpoint["num_reanalysed_routes"],
                },
                open(path, "wb"),
            )

    def terminate_workers(self):
        """
        Softly terminate the running tasks and garbage collect the workers.
        """
        if self.shared_storage_worker:
            self.shared_storage_worker.set_info.remote("terminate", True)
            self.checkpoint = ray.get(
                self.shared_storage_worker.get_checkpoint.remote()
            )
        if self.replay_buffer_worker:
            self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())

        print("\nShutting down workers...")

        self.self_route_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def infer(
        self, render=True, num_gpus=0
    ):
        """
        Inference the model in a dedicated thread.

        Args:
            render (bool): To display or not the environment. Defaults to True.
            num_gpus (int): Number of GPUs to use, 0 forces to use the CPU. Defaults to 0.
        """

        # Manage GPUs
        if 0 < self.num_gpus and self.config.selfroute_on_gpu:
            num_gpus_per_worker = self.num_gpus / (
                    self.config.num_workers * self.config.selfroute_on_gpu
            )
            if 1 < num_gpus_per_worker:
                num_gpus_per_worker = math.floor(num_gpus_per_worker)
            #else:
            #    num_gpus_per_worker = 1
        else:
            num_gpus_per_worker = 0

        # Initialize workers
        self.self_route_workers = [
            self_route.SelfRoute.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.selfroute_on_gpu else 0,
            ).remote(
                self.checkpoint,
                self.Route,
                self.config,
                self.config.seed,
                # self.config.seed + seed,
                seed,
            )
            for seed in range(self.config.num_workers)
        ]
        print("start inference")
        # Launch workers
        region_counts = [
            self_route_worker.start_infer.remote(
                render
            )
            for self_route_worker in self.self_route_workers
        ]
        while input("Enter exit to exit: ") != "exit":
            pass
        [
            self_route_worker.close_route.remote()
            for self_route_worker in self.self_route_workers
        ]

        # self_route_worker = self_route.SelfRoute.options(
        #     num_cpus=0,
        #     num_gpus=num_gpus,
        # ).remote(self.checkpoint, self.Route, self.config, numpy.random.randint(10000))
        # region_count = ray.get(
        #             self_route_worker.start_infer.remote(
        #                 render,
        #             )
        #         )
        # self_route_worker.close_route.remote()
        return region_counts

    def infer_step_by_step(
        self, render=True, num_gpus=0
    ):
        """
        Inference the model in a dedicated thread step by step.

        Args:
            render (bool): To display or not the environment. Defaults to True.
            num_gpus (int): Number of GPUs to use, 0 forces to use the CPU. Defaults to 0.
        """

        # Manage GPUs
        if 0 < self.num_gpus and self.config.selfroute_on_gpu:
            num_gpus_per_worker = self.num_gpus / (
                    self.config.num_workers * self.config.selfroute_on_gpu
            )
            if 1 < num_gpus_per_worker:
                num_gpus_per_worker = math.floor(num_gpus_per_worker)
            #else:
            #    num_gpus_per_worker = 1
        else:
            num_gpus_per_worker = 0




        # Initialize workers
        self.self_route_workers = [
            self_route.SelfRoute.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.selfroute_on_gpu else 0,
            ).remote(
                self.checkpoint,
                self.Route,
                self.config,
                self.config.seed,
                # self.config.seed + seed,
                seed,
            )
            for seed in range(self.config.num_workers)
        ]

        # Launch workers
        region_counts = [
            self_route_worker.start_infer_step_by_step.remote(
                0,
                0,
                render,
            )
            for self_route_worker in self.self_route_workers
        ]
        while input("Enter exit to exit: ") != "exit":
            pass
        [
            self_route_worker.close_route.remote()
            for self_route_worker in self.self_route_workers
        ]
        # self_route_worker = self_route.SelfRoute.options(
        #     num_cpus=0,
        #     num_gpus=num_gpus,
        # ).remote(self.checkpoint, self.Route, self.config, numpy.random.randint(10000))
        # region_count = ray.get(
        #             self_route_worker.start_infer_step_by_step.remote(
        #                 0,
        #                 0,
        #                 render,
        #             )
        #         )
        # self_route_worker.close_route.remote()
        return region_counts

    def test(
        self, render=True, num_tests=1, num_gpus=0
    ):
        """
        Test the model in a dedicated thread.

        Args:
            render (bool): To display or not the environment. Defaults to True.

            num_tests (int): Number of routes to average. Defaults to 1.

            num_gpus (int): Number of GPUs to use, 0 forces to use the CPU. Defaults to 0.
        """

        self_route_worker = self_route.SelfRoute.options(
            num_cpus=0,
            num_gpus=num_gpus,
        ).remote(self.checkpoint, self.Route, self.config, numpy.random.randint(10000))
        results = []
        for i in range(num_tests):
            print(f"Testing {i+1}/{num_tests}")
            results.append(
                ray.get(
                    self_route_worker.start_route.remote(
                        0,
                        0,
                        render,
                    )
                )
            )
        self_route_worker.close_route.remote()

        result = numpy.mean([sum(history.reward_history) for history in results])
        return result

    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
        """
        Load a model and/or a saved replay buffer.

        Args:
            checkpoint_path (str): Path to model.checkpoint or model.weights.

            replay_buffer_path (str): Path to replay_buffer.pkl
        """
        # Load checkpoint
        if checkpoint_path:
            checkpoint_path = pathlib.Path(checkpoint_path)
            self.checkpoint = torch.load(checkpoint_path)
            print(f"\nUsing checkpoint from {checkpoint_path}")

        # Load replay buffer
        if replay_buffer_path:
            replay_buffer_path = pathlib.Path(replay_buffer_path)
            with open(replay_buffer_path, "rb") as f:
                replay_buffer_infos = pickle.load(f)
            self.replay_buffer = replay_buffer_infos["buffer"]
            self.checkpoint["num_routed_steps"] = replay_buffer_infos[
                "num_routed_steps"
            ]
            self.checkpoint["num_routes"] = replay_buffer_infos[
                "num_routes"
            ]
            self.checkpoint["num_reanalysed_routes"] = replay_buffer_infos[
                "num_reanalysed_routes"
            ]

            print(f"\nInitializing replay buffer with {replay_buffer_path}")
        else:
            print(f"Using empty buffer.")
            self.replay_buffer = {}
            self.checkpoint["training_step"] = 0
            self.checkpoint["num_routed_steps"] = 0
            self.checkpoint["num_routes"] = 0
            self.checkpoint["num_reanalysed_routes"] = 0

    def diagnose_model(self, horizon):
        """
        Route only with the learned model then route the same trajectory in the real
        environment and display information.

        Args:
            horizon (int): Number of timesteps for which we collect information.
        """
        route = self.Route(self.config)
        obs = route.reset()
        dm = diagnose_model.DiagnoseModel(self.checkpoint, self.config)
        dm.compare_virtual_with_real_trajectories(obs, route, horizon)
        input("Press enter to close all plots")
        dm.close_all()


@ray.remote(num_cpus=0, num_gpus=0)
class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self, config):
        model = models.RoutingNetwork(config)
        weigths = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weigths, summary


def hyperparameter_search(
    route_module_name, parametrization, budget, parallel_experiments, num_tests
):
    """
    Search for hyperparameters by launching parallel experiments.

    Args:
        route_module_name (str): Name of the route module, it should match the name of a .py file.
        parametrization : Nevergrad parametrization, please refer to nevergrad documentation.

        budget (int): Number of experiments to launch in total.

        parallel_experiments (int): Number of experiments to launch in parallel.

        num_tests (int): Number of routes to average for evaluating an experiment.
    """
    optimizer = nevergrad.optimizers.OnePlusOne(
        parametrization=parametrization, budget=budget
    )

    running_experiments = []
    best_training = None
    try:
        # Launch initial experiments
        for i in range(parallel_experiments):
            if 0 < budget:
                param = optimizer.ask()
                print(f"Launching new experiment: {param.value}")
                xroute = XRoute(route_module_name, param.value, parallel_experiments)
                xroute.param = param
                xroute.train(False)
                running_experiments.append(xroute)
                budget -= 1

        while 0 < budget or any(running_experiments):
            for i, experiment in enumerate(running_experiments):
                if experiment and experiment.config.training_steps <= ray.get(
                    experiment.shared_storage_worker.get_info.remote("training_step")
                ):
                    experiment.terminate_workers()
                    result = experiment.test(False, num_tests=num_tests)
                    if not best_training or best_training["result"] < result:
                        best_training = {
                            "result": result,
                            "config": experiment.config,
                            "checkpoint": experiment.checkpoint,
                        }
                    print(f"Parameters: {experiment.param.value}")
                    print(f"Result: {result}")
                    optimizer.tell(experiment.param, -result)

                    if 0 < budget:
                        param = optimizer.ask()
                        print(f"Launching new experiment: {param.value}")
                        xroute = XRoute(route_module_name, param.value, parallel_experiments)
                        xroute.param = param
                        xroute.train(False)
                        running_experiments[i] = xroute
                        budget -= 1
                    else:
                        running_experiments[i] = None

    except KeyboardInterrupt:
        for experiment in running_experiments:
            if isinstance(experiment, XRoute):
                experiment.terminate_workers()

    recommendation = optimizer.provide_recommendation()
    print("Best hyperparameters:")
    print(recommendation.value)
    if best_training:
        # Save best training weights (but it's not the recommended weights)
        best_training["config"].results_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            best_training["checkpoint"],
            best_training["config"].results_path / "model.checkpoint",
        )
        # Save the recommended hyperparameters
        text_file = open(
            best_training["config"].results_path / "best_parameters.txt",
            "w",
        )
        text_file.write(str(recommendation.value))
        text_file.close()
    return recommendation.value


def load_model_menu(xroute, route_module_name, load_replay_buffer=False):
    # Configure running options
    replay_buffer_path = None
    options = ["Specify paths manually"] + sorted(
        (pathlib.Path("results") / route_module_name).glob("*/")
    )
    options.reverse()
    print()
    for i in range(len(options)):
        print(f"{i}. {options[i]}")

    choice = input("Enter a number to choose a model to load: ")
    valid_inputs = [str(i) for i in range(len(options))]
    while choice not in valid_inputs:
        choice = input("Invalid input, enter a number listed above: ")
    choice = int(choice)

    if choice == (len(options) - 1):
        # manual path option
        checkpoint_path = input(
            "Enter a path to the model.checkpoint, or ENTER if none: "
        )
        while checkpoint_path and not pathlib.Path(checkpoint_path).is_file():
            checkpoint_path = input("Invalid checkpoint path. Try again: ")
        if load_replay_buffer:
            replay_buffer_path = input(
                "Enter a path to the replay_buffer.pkl, or ENTER if none: "
            )
            while replay_buffer_path and not pathlib.Path(replay_buffer_path).is_file():
                replay_buffer_path = input("Invalid replay buffer path. Try again: ")
    else:
        checkpoint_path = options[choice] / "model.checkpoint"
        if load_replay_buffer:
            replay_buffer_path = options[choice] / "replay_buffer.pkl"

    xroute.load_model(
        checkpoint_path=checkpoint_path,
        replay_buffer_path=replay_buffer_path,
    )


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Train directly with: python xroute.py net_order
        xroute = XRoute(sys.argv[1])
        xroute.train()
    elif len(sys.argv) == 3:
        # Train or infer directly with: python xroute.py net_order '{"lr_init": 0.01}' / python xroute.py net_order '{"mode": "inference", "checkpoint": "2024-01-03--15-16-10"}'
        config = json.loads(sys.argv[2])
        xroute = XRoute(sys.argv[1], config)
        mode = "training"
        checkpoint = None
        buffer = None
        if config:
            if type(config) is dict:
                for param, value in config.items():
                    if param == "mode" and value in ["training", "inference", "inference_step_by_step"]:
                        mode = value
                    elif param == "checkpoint" and value is not None:
                        checkpoint = value
                    elif param == "buffer" and value is not None:
                        buffer = value
        if mode == "training":
            if checkpoint is not None:
                checkpoint_path = pathlib.Path("results") / sys.argv[1] / checkpoint / "model.checkpoint"
                xroute.load_model(checkpoint_path=checkpoint_path)
            if buffer is not None:
                replay_buffer_path = pathlib.Path("results") / sys.argv[1] / buffer / "replay_buffer.pkl"
                xroute.load_model(replay_buffer_path=replay_buffer_path)
            xroute.train()
        else:
            checkpoint_path = pathlib.Path("results") / sys.argv[1] / checkpoint / "model.checkpoint"
            xroute.load_model(checkpoint_path=checkpoint_path)
            if mode == "inference":
                print("Start inference...")
                xroute.infer(render=False)
            elif mode == "inference_step_by_step":
                print("Start inference step by step...")
                xroute.infer_step_by_step(render=False)
    else:
        print("\nWelcome to XRoute! Here's a list of route modules:")
        # Let user pick a route
        route_modules = [
            "net_order"
        ]
        for i in range(len(route_modules)):
            print(f"{i}. {route_modules[i]}")
        choice = "0"
        if len(route_modules) == 1:
            print(f"Automatically select the only one route module for you: {route_modules[0]}")
        else:
            choice = input("Enter a number to choose the route: ")
            valid_inputs = [str(i) for i in range(len(route_modules))]
            while choice not in valid_inputs:
                choice = input("Invalid input, enter a number listed above: ")

        # Initialize XRoute
        choice = int(choice)
        route_module_name = route_modules[choice]
        xroute = XRoute(route_module_name)

        while True:
            # Configure running options
            options = [
                "Train",
                "Load pretrained model",
                "Load pretrained model and replay buffer",
                "Inference",
                "Inference step by step",
                "Render some self routes",
                "Test the route manually",
                "Diagnose model",
                "Hyperparameter search",
                "Exit",
            ]
            print()
            for i in range(len(options)):
                print(f"{i}. {options[i]}")

            choice = input("Enter a number to choose an action: ")
            valid_inputs = [str(i) for i in range(len(options))]
            while choice not in valid_inputs:
                choice = input("Invalid input, enter a number listed above: ")
            choice = int(choice)
            if choice == 0:
                xroute.train()
            elif choice == 1:
                load_model_menu(xroute, route_module_name)
            elif choice == 2:
                load_model_menu(xroute, route_module_name, load_replay_buffer=True)
            elif choice == 3:
                xroute.infer(render=False)
            elif choice == 4:
                xroute.infer_step_by_step(render=False)
            elif choice == 5:
                xroute.test(render=True)
            elif choice == 6:
                env = xroute.Route(xroute.config)
                observation = env.reset()
                env.render()
                done = False
                while not done:
                    action = env.human_to_action()
                    observation, reward, done = env.step(action)
                    print(f"\nAction: {env.action_to_string(action)}\nReward: {reward}")
                    env.render()
                env.close()
            elif choice == 7:
                xroute.diagnose_model(30)
            elif choice == 8:
                # Define here the parameters to tune
                # Parametrization documentation: https://facebookresearch.github.io/nevergrad/parametrization.html
                xroute.terminate_workers()
                del xroute
                budget = 20
                parallel_experiments = 2
                lr_init = nevergrad.p.Log(lower=0.0001, upper=0.1)
                discount = nevergrad.p.Log(lower=0.95, upper=0.9999)
                parametrization = nevergrad.p.Dict(lr_init=lr_init, discount=discount)
                best_hyperparameters = hyperparameter_search(
                    route_module_name, parametrization, budget, parallel_experiments, 20
                )
                xroute = XRoute(route_module_name, best_hyperparameters)
            else:
                break
            print("\nDone")

    ray.shutdown()
