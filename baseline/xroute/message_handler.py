import zmq
import net_ordering_pb2 as net_ordering


class MessageHandler:
    def __init__(self, control_mq_ip=None, control_mq_port=None, data_mq_ip=None, data_mq_port=None, worker_id=0):
        self.control_context = None
        self.data_context = None
        self.control_socket = None
        self.data_socket = None

        if control_mq_ip is not None:
            self.control_mq = f"tcp://{control_mq_ip}:{control_mq_port + worker_id}"
            self.control_context = zmq.Context()
            self.control_context.setsockopt(zmq.LINGER, 0)
            self.control_socket = self.control_context.socket(zmq.REQ)
            self.control_socket.connect(self.control_mq)

        if data_mq_ip is not None:
            self.data_mq = f"tcp://{data_mq_ip}:{data_mq_port + worker_id}"
            self.data_context = zmq.Context()
            self.data_context.setsockopt(zmq.LINGER, 0)
            self.data_socket = self.data_context.socket(zmq.REP)
            self.data_socket.bind(self.data_mq)

    def send_command(self, command):
        self.control_socket.send(command)
        self.control_socket.recv()

    def send_data(self, data):
        self.data_socket.send(data)
    
    def send_action(self, action):
        message = net_ordering.Message()
        message.response.net_index = action
        self.data_socket.send(message.SerializeToString())

    def send_action_list(self, action_list):
        message = net_ordering.Message()
        message.response.net_list[:] = action_list
        self.data_socket.send(message.SerializeToString())

    def handle_message(self, message):
        data = None
        if message.HasField('request'):
            req = message.request
            data = {
                "region_coords": list(req.region_coords),
                "dimension": [req.dim_x, req.dim_y, req.dim_z],
                "grid_info": [],
                "reward_violation": req.reward_violation,
                "reward_wire_length": req.reward_wire_length,
                "reward_via": req.reward_via,
                "nets": list(req.nets),
                "graph_node_properties": [list(node_property.values) for node_property in req.graph.node_properties],
                "graph_edge_connections": [list(edge_connection.values) for edge_connection in req.graph.edge_connections],
                "is_done": req.is_done
            }

            # for node in req.nodes:
            #     node_type = 0
            #     if node.type == net_ordering.NodeType.ACCESS:
            #         node_type = node.net + 1  # 在 XRoute 中，net 是从 1 开始的
            #     elif node.type == net_ordering.BLOCKAGE:
            #         node_type = -1
            #     node_pin = -1
            #     if node.type == net_ordering.NodeType.ACCESS:
            #         node_pin = node.pin + 1  # 在 XRoute 中，pin 是从 1 开始的
            #     info = [
            #         [node.maze_x, node.maze_y, node.maze_z],
            #         [node.point_x, node.point_y, node.point_z],
            #         [int(node.is_used), node_type, node_pin],
            #     ]
            #     data[1].append(info)

        return data

    def receive_data(self):
        message_raw = self.data_socket.recv()
        message = net_ordering.Message()
        message.ParseFromString(message_raw)
        return self.handle_message(message)

    def close(self):
        if self.control_socket is not None:
            self.control_socket.close()
            self.control_socket = None
        if self.control_context is not None:
            self.control_context.destroy()
            self.control_context = None
        if self.data_socket is not None:
            self.data_socket.close()
            self.data_socket = None
        if self.data_context is not None:
            self.data_context.destroy()
            self.data_context = None


