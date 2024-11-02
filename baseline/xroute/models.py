from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch_geometric.nn import GCNConv, global_add_pool


class RoutingNetwork:
    def __new__(cls, config):
        if config.network == "fullyconnected":
            return RoutingFullyConnectedNetwork(
                config.gcn_feature_size,
                config.gcn_middle_size,
                config.gcn_out_size,
                config.action_space_size,
                config.encoding_size,
                config.fc_reward_layers,
                config.fc_value_layers,
                config.fc_policy_layers,
                config.fc_representation_layers,
                config.fc_dynamics_layers,
                config.support_size,
            )
        else:
            raise NotImplementedError(
                'The network parameter should be "fullyconnected".'
            )


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class AbstractNetwork(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action, gcn_nodes_encoding, node_num):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        s_x1 = global_add_pool(x, data.batch)
        x = self.conv2(x, edge_index).relu()
        s_x2 = global_add_pool(x, data.batch)
        # x: Node feature matrix of shape [num_nodes, out_channels]
        return x, torch.cat([s_x1, s_x2], dim=1)



##################################
######## Fully Connected #########


class RoutingFullyConnectedNetwork(AbstractNetwork):
    def __init__(
        self,
        gcn_feature_size,
        gcn_middle_size,
        gcn_out_size,
        action_space_size,
        encoding_size,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        fc_representation_layers,
        fc_dynamics_layers,
        support_size,
    ):
        super().__init__()
        assert (gcn_out_size == action_space_size)
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1

        self.gcn_network = GCN(gcn_feature_size, gcn_middle_size, gcn_out_size)

        self.representation_network = torch.nn.DataParallel(
            mlp(
                gcn_out_size * 2,
                fc_representation_layers,
                encoding_size,
            )
        )

        self.dynamics_encoded_state_network = torch.nn.DataParallel(
            mlp(
                encoding_size + action_space_size,
                fc_dynamics_layers,
                encoding_size,
            )
        )

        self.dynamics_reward_network = torch.nn.DataParallel(
            mlp(
                encoding_size,
                fc_reward_layers,
                1
            )
        )

        self.prediction_policy_network = torch.nn.DataParallel(
            mlp(
                encoding_size + action_space_size,
                fc_policy_layers,
                1
            )
        )
        self.prediction_value_network = torch.nn.DataParallel(
            mlp(
                encoding_size,
                fc_value_layers,
                1
            )
        )

    def graph_convolution(self, graph_observation):
        # graph_observation.x: all nets and their properties. Node feature matrix of shape [num_nodes, features]
        # graph_observation.edge_index: all overlap areas between two nets. Graph connectivity matrix of shape [2, num_edges]
        return self.gcn_network(graph_observation)

    def representation(self, gcn_graph_encoding):
        encoded_state = self.representation_network(gcn_graph_encoding)
        # Scale encoded state between [0, 1]
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (encoded_state - min_encoded_state) / scale_encoded_state
        return encoded_state_normalized

    def state_actions_encoding(self, encoded_state, gcn_nodes_encoding, node_num):
        encoded_states = torch.tensor([]).to(encoded_state.device)
        for i in range(len(node_num)):
            encoded_states = torch.cat([encoded_states, encoded_state[i].repeat(node_num[i], 1)], dim=0)
        return torch.cat([encoded_states, gcn_nodes_encoding], dim=1)

    def state_action_encoding(self, encoded_state, action, gcn_nodes_encoding, node_num):
        action_encoding = torch.tensor([]).to(encoded_state.device)
        index = 0
        for i in range(len(node_num)):
            action_encoding = torch.cat([action_encoding, gcn_nodes_encoding[index+action[i]:index+action[i]+1]], dim=0)
            index += node_num[i]
        return torch.cat([encoded_state, action_encoding], dim=1)

    def prediction(self, encoded_state, state_actions_encoding):
        policy_logits = self.prediction_policy_network(state_actions_encoding).T
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def dynamics(self, state_action_encoding):
        # Stack encoded_state with an action
        next_encoded_state = self.dynamics_encoded_state_network(state_action_encoding)
        reward = self.dynamics_reward_network(next_encoded_state)
        # Scale encoded state between [0, 1]
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state
        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        gcn_nodes_encoding, gcn_graph_encoding = self.graph_convolution(observation)
        encoded_state = self.representation(gcn_graph_encoding)
        node_num = []
        if len(encoded_state) == 1:
            node_num.append(observation.num_nodes)
        else:
            # batch processing
            for i in range(observation.batch_size):
                node_num.append(observation[i].num_nodes)
        state_actions_encoding = self.state_actions_encoding(encoded_state, gcn_nodes_encoding, node_num)
        policy_logits, value = self.prediction(encoded_state, state_actions_encoding)
        # reward equal to 0 for consistency
        # reward = torch.log(
        #     (
        #         torch.zeros(1, self.full_support_size)
        #         .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
        #         .repeat(len(encoded_state), 1)
        #         .to(encoded_state.device)
        #     )
        # )
        reward = torch.zeros(1, 1).repeat(len(encoded_state), 1).to(encoded_state.device)
        return (
            value,
            reward,
            policy_logits,
            encoded_state,
            gcn_nodes_encoding,
            node_num
        )

    def recurrent_inference(self, encoded_state, action, gcn_nodes_encoding, node_num):
        state_action_encoding = self.state_action_encoding(encoded_state, action, gcn_nodes_encoding, node_num)
        next_encoded_state, reward = self.dynamics(state_action_encoding)
        state_actions_encoding = self.state_actions_encoding(next_encoded_state, gcn_nodes_encoding, node_num)
        policy_logits, value = self.prediction(next_encoded_state, state_actions_encoding)
        return value, reward, policy_logits, next_encoded_state


###### End Fully Connected #######
##################################


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)


def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=1)
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x


def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits








