import os
import sys
import time
from multiprocessing import cpu_count

import zmq
from jinja2 import Environment, FileSystemLoader

from dispatcher import Dispatcher

jinja_env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), 'templates')))


class Trainer:
    def __init__(self, openroad_executable, dump_dir, port_control=5555, port_to_alg=6666):
        self.port_control = port_control
        self.port_to_alg = port_to_alg
        self.openroad_executable = openroad_executable
        self.dump_dir = dump_dir

        self.current_region_index = 0
        self.worker_dir_names = []
        self.load_regions_info()

        self.zmq_context = zmq.Context()
        self.zmq_context.setsockopt(zmq.LINGER, 0)
        self.socket_control = self.zmq_context.socket(zmq.REP)
        self.socket_control.bind(f'tcp://*:{self.port_control}')

        self.dispatcher = None

    # port to receive request from OpenROAD in step mode
    @property
    def port_from_or_mixer(self):
        return '2' + str(self.port_to_alg)

    @property
    def current_region(self):
        return self.worker_dir_names[self.current_region_index]

    def load_regions_info(self):
        dir_items = os.listdir(self.dump_dir)
        self.worker_dir_names = [name for name in dir_items if name.startswith("workerx")]

    def kill_dispatcher(self):
        if self.dispatcher is not None:
            self.dispatcher.terminate()
            self.dispatcher = None
            os.system(f'ps aux | grep \'_{port_to_alg}.tcl\' | grep -v grep | awk \'{{print $2}}\' | xargs -r kill -9')
            time.sleep(0.01)

    def generate_tcl_script(self):
        template_mixer = jinja_env.get_template('mixer.tcl')
        script = template_mixer.render(thread_count=cpu_count(),
                                         api_address=f'127.0.0.1:{self.port_from_or_mixer}',
                                         dump_dir=self.dump_dir,
                                         worker_dir=self.current_region)
        with open(f'tmp/mixer_{self.port_to_alg}.tcl', 'w') as f:
            f.write(script)

        print('Scripts generated.\n')

    def start_server(self):
        print('Waiting for request...')

        if not os.path.exists('tmp'):
            os.mkdir('tmp')

        while True:
            message_control = self.socket_control.recv()
            self.socket_control.send(b'ok')

            self.kill_dispatcher()

            if message_control == b'reset':
                print('Reset.')
                self.current_region_index = 0
            elif message_control == b'jump':
                print('Jump.')

                if self.current_region_index == len(self.worker_dir_names) - 1:
                    print('Jump to the first region.')
                    self.current_region_index = 0
                else:
                    self.current_region_index += 1

            print(f'Current region: {self.current_region_index} - {self.current_region}')

            self.generate_tcl_script()

            self.dispatcher = Dispatcher(openroad_executable=self.openroad_executable,
                                         port_to_alg=self.port_to_alg,
                                         port_from_or_mixer=self.port_from_or_mixer)
            self.dispatcher.start()


if __name__ == "__main__":
    openroad_executable = ''
    dump_dir = ''

    if len(sys.argv) >= 3:
        port_control = int(sys.argv[1])
        port_to_alg = int(sys.argv[2])

        if len(sys.argv) == 4:
            openroad_executable = ''
            dump_dir = ''
        elif len(sys.argv) > 4:
            print('Wrong number of arguments.')
            exit(1)
    elif len(sys.argv) == 1:
        port_control = 5555
        port_to_alg = 6666
    else:
        print('Wrong number of arguments.')
        exit(1)

    trainer = Trainer(
        openroad_executable=openroad_executable,
        dump_dir=dump_dir,
        port_control=port_control,
        port_to_alg=port_to_alg)
    trainer.start_server()
