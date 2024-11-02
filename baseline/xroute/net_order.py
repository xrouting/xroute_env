import datetime
import pathlib
import torch
import numpy
from abstract_route import AbstractRoute
from message_handler import MessageHandler


class RouteConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and route
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Routing environment
        self.control_mq_ip = "127.0.0.1"
        self.control_mq_port = 6667
        self.data_mq_ip = "*"
        self.data_mq_port = 5556
        self.infer_mq_ip = "*"
        self.infer_mq_port = 10888
        self.mode = "training"  # "training" or "inference" or "inference_step_by_step" mode to run
        self.checkpoint = ''  # Path to a checkpoint to load
        self.buffer = ''  # Path to a replay buffer to load

        ### Route
        self.action_space = list(range(10))  # Fixed list of all possible actions. for example, list(range(10)), for diagnose model only.
        self.action_space_size = 11  # Feature size of actions, keep the same with gcn_feature_size

        ### Self-Route
        self.one_env_instance_only = True   # Only one environment instance, number_workers = 1 and set self_route_per_test
        self.num_workers = 32  # Number of simultaneous threads/workers self-routing to feed the replay buffer, work with routes_per_region
        self.self_route_per_test = 5   # Number of self routes after 1 test with the latest model, None for self route only
        self.selfroute_on_gpu = True
        self.max_moves = 50  # Maximum number of moves if route is not finished before
        self.num_simulations = 64  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "fullyconnected"  # "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Graph Convolution Network
        self.gcn_feature_size = 11  # Feature size of nodes in graph convolution network
        self.gcn_middle_size = 11  # Middle size of nodes in graph convolution network
        self.gcn_out_size = 11  # Graph representation size of nodes in graph convolution network

        # Fully Connected Network
        self.encoding_size = 64
        self.fc_representation_layers = [64,64]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64,64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64,64]  # Define the hidden layers in the reward network
        self.fc_value_layers = [64,64]  # Define the hidden layers in the value network
        self.fc_policy_layers = [64,64]  # Define the hidden layers in the policy network

        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[0] / "results" / pathlib.Path(
            __file__).stem / datetime.datetime.now().strftime(
            "%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 30000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 512  # Number of parts of routes to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-routing
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available
        self.batch_loss_function = "Mean"  # "Mean" or "Sum"
        self.routes_per_region = 100   # None for always routing in one region, or a number bigger than 0 for routing times per region
        self.reset_region = True    # True to begin training on the first region and False to continue training on the current region

        self.optimizer = "Adam"  # "Adam" or "SGD" or "RMSprop".
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.01  # Initial learning rate
        self.lr_decay_rate = 0.9  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 5000

        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-route routes to keep in the replay buffer
        self.expert_replay_buffer_size = 2000
        self.num_unroll_steps = 5  # Number of route moves to keep for every batch element
        self.td_steps = 5  # Number of steps in the future to take into account for calculating the target value
        self.PER = False  # Prioritized Replay, select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        # self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.use_last_model_value = True
        self.reanalyse_on_gpu = True

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_route_delay = 0  # Number of seconds to wait after each route
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self routed step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on
        # Ray debug
        # self.ray_debug = True
        self.ray_debug = False
        
        # action perfer
        self.num_action = 10
        self.epslion = 0                        #-1代表原始xroute


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Route(AbstractRoute):
    """
    Route wrapper.
    """
    
    def __init__(self, config, seed=None, worker_id=0):
        # Fix random generator seed
        self.reset_region = config.reset_region
        self.routes_per_region = config.routes_per_region
        self.routes_in_region = 0
        self.observation = None
        self.legal_nets = None
        self.net_space = None
        self.reward = None
        self.route_name = None
        self.reward_change_times = -1
        if seed is None:
            seed = config.seed
        if seed is not None:
            # numpy.random.seed(seed)
            # torch.manual_seed(seed)

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

        self.worker_id = worker_id
        print(f"XRoute {worker_id} for net ordering is running in {config.mode} mode...")
        if config.mode == "training":
            self.msg_handler = MessageHandler(config.control_mq_ip, config.control_mq_port, config.data_mq_ip, config.data_mq_port, worker_id)
        else:
            self.msg_handler = MessageHandler(None, None, config.infer_mq_ip, config.infer_mq_port, worker_id)

    def force_terminate(self):
        self.msg_handler.send_action(-1)

    def step(self, action):
        """
        Apply action to the route.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the route has ended.
        """
        self.msg_handler.send_action(action)
        # print(f"Instance {self.worker_id} send {action}")
        data = self.msg_handler.receive_data()
        # print(f"Instance {self.worker_id} receive {data['nets']}")
        self.legal_nets = set(data['nets'])
        done = data['is_done']
        observation = {"graph_node_properties": data['graph_node_properties'], "graph_edge_connections": data['graph_edge_connections']}
        reward_violation = data['reward_violation']
        reward_wire_length = data['reward_wire_length']
        reward_via = data['reward_via']
        reward = (0.5 * reward_wire_length + 4 * reward_via + 500 * reward_violation) / 1000
        print(f'Instance {self.worker_id} action {action} reward_wire_length: {reward_wire_length}, reward_via: {reward_via}, reward_violation: {reward_violation}, total_reward: {reward}')
        if reward != 0:
            self.reward_change_times += 1
        if done:
            self.msg_handler.send_data(b'\0')
            print(
                f"Instance {self.worker_id} no. {self.routes_in_region} routing on {self.route_name}: reward change {self.reward_change_times} times.")
        return observation, reward, done

    def step_inference(self, action_list):
        """
        Apply actions to the route.

        Args:
            action_list : actions of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the route has ended.
        """
        self.msg_handler.send_action_list(action_list)
        print(f"Instance {self.worker_id} send action list:{action_list}")
        return None

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the route have to be able to handle one of returned actions.

        For complex route where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.legal_nets
    

    def legal_actions_with_window(self,index,window_size):

        raise NotImplementedError
        

    def get_action_space(self):
        return self.net_space

    def reset(self):
        """Reset the route for a new route.
        Returns:
            observation: graph data including node properties and edge connection.
        """
        if self.reset_region:
            command = b'reset'
            self.reset_region = False
            self.routes_in_region = 1
        elif self.reward_change_times == 0 or (self.routes_per_region is not None and self.routes_in_region >= self.routes_per_region):
            command = b'jump'
            self.routes_in_region = 1
        else:
            command = b'initial'
            self.routes_in_region += 1
        done = True
        while done:
            self.msg_handler.send_command(command)
            print(f"Instance {self.worker_id} send {str(command)} to routing environment...")
            data = self.msg_handler.receive_data()
            self.route_name = str(data['region_coords'])
            print(f"Instance {self.worker_id} receive {data['nets']} from region {self.route_name}")
            self.reward_change_times = 0
            done = data['is_done']
            self.net_space = data['nets']
            self.legal_nets = set(data['nets'])
            observation = {"graph_node_properties": data['graph_node_properties'], "graph_edge_connections": data['graph_edge_connections']}
            violation = data['reward_violation']
            wire_length = data['reward_wire_length']
            via = data['reward_via']
            score = 0.5 * wire_length + 4 * via + 500 * violation
            print(
                f"Instance {self.worker_id} in region: {self.route_name} baseline: actions: {len(self.net_space)}, wire_length: {wire_length}, via: {via}, violation: {violation}, total_score: {score}")
            if done:
                self.msg_handler.send_data(b'\0')
                print(f"Instance {self.worker_id} in region: {self.route_name} finish routing")
                command = b'jump'
                self.routes_in_region = 1
            elif len(self.net_space) > 0 and len(self.net_space) != max(self.net_space) + 1:
                print(f"Instance {self.worker_id} error: number of nets not equal to max net_id + 1, net space: {self.net_space}, region: {self.route_name}")
                self.force_terminate()
                done = True
                command = b'jump'
                self.routes_in_region = 1

        return observation

    def reset_inference(self):
        """Reset the route for a new route inference.
        Returns:
            observation: graph data including node properties and edge connection.
        """
        done = True
        observation = None
        while done:
            data = self.msg_handler.receive_data()
            self.route_name = str(data['region_coords'])
            print(f"Instance {self.worker_id} receive {data['nets']} from region {self.route_name}")
            done = data['is_done']
            if done:
                self.msg_handler.send_data(b'\0')
                print(f"Instance {self.worker_id} in region {self.route_name} has finished routing")
            else:
                self.net_space = data['nets']
                #要求data['net']是顺序的，不然列表转字典后续遍历会有问题
                #print("datanets:",data['nets'])
                self.legal_nets = set(data['nets'])
                observation = {"graph_node_properties": data['graph_node_properties'], "graph_edge_connections": data['graph_edge_connections']}
        return observation

    def close(self):
        """
        Properly close the route.
        """
        self.msg_handler.close()
        return None

    def render(self):
        """
        Display the route observation.
        """
        return None

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return str(action_number)
