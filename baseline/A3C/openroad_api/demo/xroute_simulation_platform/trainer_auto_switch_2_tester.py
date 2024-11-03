# coding=utf-8

import json
import os
import random
import sys

import zmq

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(parent_dir)

import proto.net_ordering_pb2 as net_ordering

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect('tcp://localhost:5555')
socket.send(b'reset')

socket = context.socket(zmq.REP)
socket.bind('tcp://*:6666')

while True:
    message_raw = socket.recv()
    message = net_ordering.Message()
    message.ParseFromString(message_raw)

    if message.HasField('request'):
        req = message.request
        message = net_ordering.Message()

        if len(req.nets) > 0:
            message.response.net_list.extend(req.nets)
            random.shuffle(message.response.net_list)
            socket.send(message.SerializeToString())
        elif len(req.xroute) > 0:
            print(req.openroad)
            print(req.xroute)
            count_map = {str(int(key)+1): value for key, value in json.loads(req.count_map).items()}
            print(count_map)
            metrics_delta = {str(int(key)+1): value for key, value in json.loads(req.metrics_delta).items()}
            print(metrics_delta)
            sys.exit(0)
        else:
            print('Invalid message')
            sys.exit(1)

    else:
        print('Invalid message')
        sys.exit(1)
