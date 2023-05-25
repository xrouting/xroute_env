# coding=utf-8
import json
import os
import random
import sys
import time

import zmq

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(parent_dir)

import proto.net_ordering_pb2 as net_ordering


# 请求初始化环境
# context = zmq.Context()
# socket = context.socket(zmq.REQ)
# socket.connect('tcp://127.0.0.1:6666')
# socket.send(b'init')

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://*:5555')

net_set = list()

while True:
    message_raw = socket.recv()

    # 解析消息
    message = net_ordering.Message()
    message.ParseFromString(message_raw)

    if message.HasField('request'):
        req = message.request

        if req.is_done:
            net_set = list()
            print('\033[31mDone.\033[0m\n')
            socket.send(b'\0')

        else:
            data = [
                [req.dim_x, req.dim_y, req.dim_z],
                [],
                [req.reward_violation, req.reward_wire_length, req.reward_via],
            ]

            for node in req.nodes:
                node_type = 0
                if node.type == net_ordering.ACCESS:
                    node_type = node.net + 1  # XRoute 的 net 是从 1 开始的，所以先加上去，等下再减回去
                elif node.type == net_ordering.BLOCKAGE:
                    node_type = -1

                node_pin = -1
                if node.type == net_ordering.NodeType.ACCESS:
                    node_pin = node.pin + 1  # XRoute 的 pin 是从 1 开始的

                info = [
                    [node.maze_x, node.maze_y, node.maze_z],
                    [node.point_x, node.point_y, node.point_z],
                    [int(node.is_used), node_type, node_pin],
                ]

                data[1].append(info)

            # dataDir = 'logs'
            # if not os.path.exists(dataDir):
            #     os.makedirs(dataDir)
            # with open(f'{dataDir}/grid_graph_{round(time.time() * 1000)}.json', 'w') as f:
            #     f.write(json.dumps(data, separators=[',', ':']))

            print(req.nets)
            idx = random.randint(0, len(req.nets) - 1)
            netIndex = req.nets[idx] + 1
            print(f'\033[31m{netIndex}\033[0m')

            # netIndex = 1
            # is_input = False
            # while not is_input:
            #     input_str = input('Input a net index: ')
            #
            #     # 校验
            #     if not input_str.isdigit():
            #         print('net index must be a number.')
            #     elif int(input_str) <= 0:
            #         print('net index must be greater than 0.')
            #     else:
            #         netIndex = int(input_str)
            #         is_input = True

            # 回复 netIndex
            message = net_ordering.Message()
            message.response.net_index = netIndex - 1  # OpenROAD 的 net 是从 0 开始的，所以返回时记得减回去
            socket.send(message.SerializeToString())

    else:
        print('Unknown message type.')
