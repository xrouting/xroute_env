import copy
import time

import numpy
import ray
import torch

import models
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


@ray.remote
class ReplayBuffer:
    """
    Class which run in a dedicated thread to store routes and generate batch.
    """

    def __init__(self, initial_checkpoint, initial_buffer, config):
        self.config = config
        self.buffer = copy.deepcopy(initial_buffer)
        self.num_routes = initial_checkpoint["num_routes"]
        self.num_routed_steps = initial_checkpoint["num_routed_steps"]
        self.total_samples = sum(
            [len(route_history.root_values) for route_history in self.buffer.values()]
        )
        if self.total_samples != 0:
            print(
                f"Replay buffer initialized with {self.total_samples} samples ({self.num_routes} routes).\n"
            )

        # Fix random generator seed
        #numpy.random.seed(self.config.seed)

        seed = self.config.seed
        # random.seed(seed)
        # os.environ['PYTHONHASHSEED'] = str(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Remove randomness (may be slower on Tesla GPUs)
        # https://pytorch.org/docs/stable/notes/randomness.html
        #if seed == 0:
        #    torch.backends.cudnn.deterministic = True
        #    torch.backends.cudnn.benchmark = False

    def save_route(self, route_history, shared_storage=None):
        if self.config.PER:
            if route_history.priorities is not None:
                # Avoid read only array when loading replay buffer from disk
                route_history.priorities = numpy.copy(route_history.priorities)
            else:
                # Initial priorities for the prioritized replay, the smaller, the better
                priorities = []
                for i, root_value in enumerate(route_history.root_values):
                    priority = (
                        numpy.abs(
                            root_value - self.compute_target_value(route_history, i)
                        )
                        ** self.config.PER_alpha
                    )
                    priorities.append(priority)

                route_history.priorities = numpy.array(priorities, dtype="float32")
                # See definition of self.PER: select in priority the elements in the replay buffer which are unexpected for the network
                # Confirm the route_priority is the max priority, and priority is determined by root_value-target_value
                route_history.route_priority = numpy.max(route_history.priorities)

        # self.buffer is a dict
        self.buffer[self.num_routes] = route_history
        self.num_routes += 1
        self.num_routed_steps += len(route_history.root_values)
        self.total_samples += len(route_history.root_values)

        # delete the oldest route_history in buffer
        if self.config.replay_buffer_size < len(self.buffer):
            del_id = self.num_routes - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]

        if shared_storage:
            shared_storage.set_info.remote("num_routes", self.num_routes)
            shared_storage.set_info.remote("num_routed_steps", self.num_routed_steps)

    def get_buffer(self):
        return self.buffer

    def get_batch(self):
        (
            index_batch,
            observation_batch,
            action_batch,
            reward_batch,
            value_batch,
            policy_batch,
            gradient_scale_batch,
        ) = ([], [], [], [], [], [], [])
        weight_batch = [] if self.config.PER else None

        for route_id, route_history, route_prob in self.sample_n_routes(
            self.config.batch_size
        ):
            route_pos, pos_prob = self.sample_position(route_history)

            values, rewards, policies, actions = self.make_target(
                route_history, route_pos
            )

            index_batch.append([route_id, route_pos])
            observation_batch.append(
                route_history.get_route_observation(route_pos)
            )
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            gradient_scale_batch.append(
                [
                    min(
                        self.config.num_unroll_steps,
                        len(route_history.action_history) - route_pos,
                    )
                ]
                * len(actions)
            )
            if self.config.PER:
                weight_batch.append(1 / (self.total_samples * route_prob * pos_prob))

        if self.config.PER:
            weight_batch = numpy.array(weight_batch, dtype="float32") / max(
                weight_batch
            )

        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # weight_batch: batch
        # gradient_scale_batch: batch, num_unroll_steps+1
        return (
            index_batch,
            (
                observation_batch,
                action_batch,
                value_batch,
                reward_batch,
                policy_batch,
                weight_batch,
                gradient_scale_batch,
            ),
        )

    def sample_route(self, force_uniform=False):
        """
        Sample route from buffer either uniformly or according to some priority.
        """
        route_prob = None
        if self.config.PER and not force_uniform:
            route_probs = numpy.array(
                [route_history.route_priority for route_history in self.buffer.values()],
                dtype="float32",
            )
            route_probs /= numpy.sum(route_probs)
            route_index = numpy.random.choice(len(self.buffer), p=route_probs)
            route_prob = route_probs[route_index]
        else:
            route_index = numpy.random.choice(len(self.buffer))
        route_id = self.num_routes - len(self.buffer) + route_index

        return route_id, self.buffer[route_id], route_prob

    def sample_n_routes(self, n_routes, force_uniform=False):
        if self.config.PER and not force_uniform:
            route_id_list = []
            route_probs = []
            for route_id, route_history in self.buffer.items():
                route_id_list.append(route_id)
                route_probs.append(route_history.route_priority)
            route_probs = numpy.array(route_probs, dtype="float32")
            route_probs /= numpy.sum(route_probs)
            route_prob_dict = dict(
                [(route_id, prob) for route_id, prob in zip(route_id_list, route_probs)]
            )
            selected_routes = numpy.random.choice(route_id_list, n_routes, p=route_probs)
        else:
            selected_routes = numpy.random.choice(list(self.buffer.keys()), n_routes)
            route_prob_dict = {}
        ret = [
            (route_id, self.buffer[route_id], route_prob_dict.get(route_id))
            for route_id in selected_routes
        ]
        return ret

    def sample_position(self, route_history, force_uniform=False):
        """
        Sample position from route either uniformly or according to some priority.
        """
        position_prob = None
        if self.config.PER and not force_uniform:
            position_probs = route_history.priorities / sum(route_history.priorities)
            position_index = numpy.random.choice(len(position_probs), p=position_probs)
            position_prob = position_probs[position_index]
        else:
            position_index = numpy.random.choice(len(route_history.root_values))

        return position_index, position_prob

    def update_route_history(self, route_id, route_history):
        # The element could have been removed since its selection and update
        if next(iter(self.buffer)) <= route_id:
            if self.config.PER:
                # Avoid read only array when loading replay buffer from disk
                route_history.priorities = numpy.copy(route_history.priorities)
            self.buffer[route_id] = route_history

    def update_priorities(self, priorities, index_info):
        """
        Update route and position priorities with priorities calculated during the training.
        """
        for i in range(len(index_info)):
            route_id, route_pos = index_info[i]

            # The element could have been removed since its selection and training
            if next(iter(self.buffer)) <= route_id:
                # Update position priorities
                priority = priorities[i, :]
                start_index = route_pos
                end_index = min(
                    route_pos + len(priority), len(self.buffer[route_id].priorities)
                )
                self.buffer[route_id].priorities[start_index:end_index] = priority[
                    : end_index - start_index
                ]

                # Update route priorities
                self.buffer[route_id].route_priority = numpy.max(
                    self.buffer[route_id].priorities
                )

    def compute_target_value(self, route_history, index):
        # The value target is the discounted root value of the search tree td_steps into the
        # future, plus the discounted sum of all rewards until then.
        bootstrap_index = index + self.config.td_steps
        if bootstrap_index < len(route_history.root_values):
            root_values = (
                route_history.root_values
                if route_history.reanalysed_predicted_root_values is None
                else route_history.reanalysed_predicted_root_values
            )
            last_step_value = root_values[bootstrap_index]
            value = last_step_value * self.config.discount**self.config.td_steps
        else:
            value = 0
        for i, reward in enumerate(
            route_history.reward_history[index + 1 : bootstrap_index + 1]
        ):
            value += reward * self.config.discount**i
        return value

    def make_target(self, route_history, state_index):
        """
        Generate targets for every unroll steps.
        """
        target_values, target_rewards, target_policies, actions = [], [], [], []
        for current_index in range(
            state_index, state_index + self.config.num_unroll_steps + 1
        ):
            value = self.compute_target_value(route_history, current_index)

            if current_index < len(route_history.root_values):
                target_values.append(value)
                target_rewards.append(route_history.reward_history[current_index])
                target_policies.append(route_history.child_visits[current_index])
                actions.append(route_history.action_history[current_index])
            elif current_index == len(route_history.root_values):
                target_values.append(0)
                target_rewards.append(route_history.reward_history[current_index])
                # Uniform policy
                target_policies.append(
                    [
                        # 1 / len(route_history.child_visits[0])
                        0
                        for _ in range(len(route_history.child_visits[0]))
                    ]
                )
                actions.append(route_history.action_history[current_index])
            else:
                # States past the end of routes are treated as absorbing states
                target_values.append(0)
                target_rewards.append(0)
                # Uniform policy
                target_policies.append(
                    [
                        # 1 / len(route_history.child_visits[0])
                        0
                        for _ in range(len(route_history.child_visits[0]))
                    ]
                )
                actions.append(numpy.random.choice(route_history.action_space))

        return target_values, target_rewards, target_policies, actions


@ray.remote
class Reanalyse:
    """
    Class which run in a dedicated thread to update the replay buffer with fresh information.
    """

    def __init__(self, initial_checkpoint, config):
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

        # Initialize the network
        self.model = models.RoutingNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.reanalyse_on_gpu else "cpu"))
        self.model.eval()

        self.num_reanalysed_routes = initial_checkpoint["num_reanalysed_routes"]

    def reanalyse(self, replay_buffer, shared_storage):
        while ray.get(shared_storage.get_info.remote("num_routes")) < 1:
            time.sleep(0.1)

        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

            route_id, route_history, _ = ray.get(
                replay_buffer.sample_route.remote(force_uniform=True)
            )

            # Use the last model to provide a fresher, stable n-step value
            if self.config.use_last_model_value:
                # observations = numpy.array(
                #     [
                #         route_history.get_route_observation(i)
                #         for i in range(len(route_history.root_values))
                #     ]
                # )
                # observations = (
                #     torch.tensor(observations)
                #     .float()
                #     .to(next(self.model.parameters()).device)
                # )
                device = next(self.model.parameters()).device
                observation_batch = [route_history.get_route_observation(i) for i in range(len(route_history.root_values))]
                graphs = []
                for i in range(len(observation_batch)):
                    x = torch.tensor(observation_batch[i]['graph_node_properties'], dtype=torch.float).to(device)
                    edge_index = torch.tensor(observation_batch[i]['graph_edge_connections'], dtype=torch.long).to(device)
                    if len(edge_index):
                        edge_index = torch.cat((edge_index, edge_index[:, [1, 0]]), dim=0).T
                    else:
                        edge_index = torch.tensor([[], []], dtype=torch.long).to(device)
                    graphs.append(Data(x=x, edge_index=edge_index))
                loader = DataLoader(graphs, batch_size=len(graphs))
                graphs_observation = next(iter(loader))

                values = self.model.initial_inference(graphs_observation)[0]
                # values = models.support_to_scalar(
                #     self.model.initial_inference(graphs_observation)[0],
                #     self.config.support_size,
                # )
                route_history.reanalysed_predicted_root_values = (
                    torch.squeeze(values).detach().cpu().numpy()
                )

            replay_buffer.update_route_history.remote(route_id, route_history)
            self.num_reanalysed_routes += 1
            shared_storage.set_info.remote(
                "num_reanalysed_routes", self.num_reanalysed_routes
            )
