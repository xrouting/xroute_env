import os
import random
import sys

import ipdb
import zmq

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(parent_dir)

import proto.net_ordering_pb2 as net_ordering

if len(sys.argv) != 3:
    port_control = 5555
    port_to_alg = 6666
else:
    port_control = int(sys.argv[1])
    port_to_alg = int(sys.argv[2])

context = zmq.Context()
context.setsockopt(zmq.LINGER, 0)
socket_req = context.socket(zmq.REQ)
socket_req.connect(f'tcp://localhost:{port_control}')
socket_req.send(b'reset')
socket_req.recv()

socket_rep = context.socket(zmq.REP)
socket_rep.bind(f'tcp://*:{port_to_alg}')

count = 0
while True:
    message_raw = socket_rep.recv()
    message = net_ordering.Message()
    message.ParseFromString(message_raw)

    # ipdb.set_trace()

    if message.HasField('request'):
        req = message.request
        print(req.nets)
        print([req.reward_violation, req.reward_wire_length, req.reward_via])

        node_props = [list(i.values) for i in req.graph.node_properties]
        print(node_props)

        message = net_ordering.Message()

        if len(req.nets) == 0:
            socket_rep.send(b'ok')
            print('Done.')
            # sys.exit(0)

        else:
            random_index = random.randint(0, len(req.nets) - 1)
            net_index = req.nets[random_index]

            # count+= 1
            # if count == 3:
            #     print('Tell trainer to stop immediately.')
            #     net_index = -1

            message.response.net_index = net_index
            # random.shuffle(req.nets)
            # message.response.net_list[:] = list(req.nets)

            print(f'Selected net index: {message.response.net_index}.\n')
            socket_rep.send(message.SerializeToString())

    else:
        print('Invalid message')
        sys.exit(1)
