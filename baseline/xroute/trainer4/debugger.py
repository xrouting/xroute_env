import os
import random
import sys

import ipdb
import zmq

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(parent_dir)

import proto.net_ordering_pb2 as net_ordering


context = zmq.Context()
context.setsockopt(zmq.LINGER, 0)
socket = context.socket(zmq.REP)
socket.bind('tcp://*:5555')

net_set = []

while True:
    message_raw = socket.recv()
    message = net_ordering.Message()
    message.ParseFromString(message_raw)

    if message.HasField('request'):
        req = message.request

        if req.is_done:
            net_set = []
            print('\033[31mDone.\033[0m\n')
            socket.send(b'\0')

        else:
            print(req.region_coords)
            print(req.nets)

            message = net_ordering.Message()
            message.response.net_list[:] = req.nets #[4, 0, 1, 2, 3]
            #random.shuffle(message.response.net_list)
            socket.send(message.SerializeToString())
    else:
        print('Unknown message type.')
