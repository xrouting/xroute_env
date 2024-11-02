import os
import sys
from multiprocessing import Process, Queue

import zmq

from mixer import Mixer

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(parent_dir)

import proto.net_ordering_pb2 as net_ordering


class Dispatcher(Process):
    def __init__(self, openroad_executable, port_to_alg, port_from_or_mixer):
        super().__init__()

        self.openroad_executable = openroad_executable
        self.port_to_alg = port_to_alg
        self.port_from_or_mixer = port_from_or_mixer

        self.zmq_context = None
        self.socket_to_alg = None

    def __del__(self):
        if self.socket_to_alg is not None:
            self.socket_to_alg.close()
            self.socket_to_alg = None

        if self.zmq_context is not None:
            self.zmq_context.destroy()
            self.zmq_context = None

        print('Dispatcher terminated.\n')

    def run(self):
        self.zmq_context = zmq.Context()
        self.zmq_context.setsockopt(zmq.LINGER, 0)
        self.socket_to_alg = self.zmq_context.socket(zmq.REQ)
        self.socket_to_alg.connect(f'tcp://127.0.0.1:{self.port_to_alg}')

        unrouted_nets = []
        routed_nets = []
        init_metrics = [0, 0, 0]    # sometime metrics will not be 0 at the beginning
        last_metrics = [0, 0, 0]
        last_result = None

        while True:
            mixer = self.start_mixer()
            result = mixer.get_observation()

            if last_result is None:
                # Consider case of empty region
                if len(result.nets) > 0:
                    unrouted_nets = list(result.nets)
                    init_metrics = [result.reward_violation, result.reward_wire_length, result.reward_via]
                    print(f'Applying default order: {result.nets}')
                    mixer.set_net_list(result.nets)
                    result = mixer.get_result()    # Get result of default order

                    # After ack, mixer will exit, so we need to restart it
                    mixer.ack()
                    mixer = None
                    mixer = self.start_mixer()
                    mixer.get_observation() # don't pass result to varaible `result`
            else:
                result = last_result

            # Check if all nets are routed
            result.is_done = True if len(unrouted_nets) == 0 else False

            # Convert metrics to rewards
            raw_metrics = [result.reward_violation, result.reward_wire_length, result.reward_via]
            current_metrics = [a - b for a, b in zip(raw_metrics, init_metrics)]
            delta_metrics = [b - a for a, b in zip(current_metrics, last_metrics)]
            print(f'Delta metrics: {delta_metrics}')
            result.reward_violation = delta_metrics[0]
            result.reward_wire_length = delta_metrics[1]
            result.reward_via = delta_metrics[2]
            last_metrics = current_metrics

            # Add is_routed to node property
            for index, node_property in enumerate(result.graph.node_properties):
                node_property.values[3] = 1 if index in routed_nets else 0

            # Pass observation to algorithm
            message = net_ordering.Message()
            result.nets[:] = unrouted_nets
            print(f'Unrouted nets: {unrouted_nets}')
            message.request.CopyFrom(result)
            self.socket_to_alg.send(message.SerializeToString())

            if len(unrouted_nets) == 0:
                self.socket_to_alg.recv()
                mixer = None
                break

            # Receive selected net index from algorithm
            raw_message = self.socket_to_alg.recv()
            message = net_ordering.Message()
            message.ParseFromString(raw_message)

            # Save selected net index locally
            selected_net_index = message.response.net_index
            # Algorithm will send -1 if it wants to stop immediately
            if selected_net_index == -1:
                print('Algorithm wants to stop immediately.')
                mixer = None
                break

            routed_nets.append(selected_net_index)
            unrouted_nets.remove(selected_net_index)

            # Pass selected net to mixer
            mixer.set_net_list(routed_nets + unrouted_nets)
            print(f'Applying order: {routed_nets + unrouted_nets}')

            # Receive result from mixer
            last_result = mixer.get_result()
            mixer.ack()
            mixer = None

    def start_mixer(self):
        mixer = Mixer(openroad_executable=self.openroad_executable,
                          tcl_script=f'tmp/mixer_{self.port_to_alg}.tcl',
                          port_from_or_mixer=self.port_from_or_mixer)
        mixer.start()
        return mixer