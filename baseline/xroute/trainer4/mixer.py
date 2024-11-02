import os
import subprocess
import sys

import zmq

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(parent_dir)

import proto.net_ordering_pb2 as net_ordering


class Mixer:
    def __init__(self, openroad_executable, tcl_script, port_from_or_mixer):
        super().__init__()
        self.openroad_executable = openroad_executable
        self.tcl_script = tcl_script
        self.port_from_or_mixer = port_from_or_mixer

        self.zmq_context = None
        self.socket_from_or_mixer = None

        self.process = None

    def __del__(self):
        if self.process is not None:
            self.process.terminate()
            self.process = None
            print('Mixer terminated.\n')

        if self.socket_from_or_mixer is not None:
            self.socket_from_or_mixer.close()
            self.socket_from_or_mixer = None

        if self.zmq_context is not None:
            self.zmq_context.destroy()
            self.zmq_context = None

    def start(self):
        print('Starting mixer...')

        self.zmq_context = zmq.Context()
        self.zmq_context.setsockopt(zmq.LINGER, 0)
        self.socket_from_or_mixer = self.zmq_context.socket(zmq.REP)
        self.socket_from_or_mixer.bind(f'tcp://*:{self.port_from_or_mixer}')

        self.process = subprocess.Popen([self.openroad_executable, '-exit', self.tcl_script], stdout=subprocess.PIPE)

    def get_observation(self):
        raw_message = self.socket_from_or_mixer.recv()
        message = net_ordering.Message()
        message.ParseFromString(raw_message)
        return message.request

    # for semantic consistency
    def get_result(self):
        return self.get_observation()

    def set_net_list(self, net_list):
        message = net_ordering.Message()
        message.response.net_list[:] = net_list
        self.socket_from_or_mixer.send(message.SerializeToString())

    # acknowledge the result
    def ack(self):
        self.socket_from_or_mixer.send(b'ok')
