import multiprocessing
import os
import random
import subprocess
import threading
import time

from jinja2 import Environment, FileSystemLoader

jinja_env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), 'templates')))

subprocesses = {}


class OpenRoadTask(threading.Thread):
    def __init__(self,
                 task_mode='normal',
                 task_timeout=86400 * 365,
                 testcase_name='ispd18_sample',
                 custom_strategies=[],
                 parallel_workers=multiprocessing.cpu_count(),
                 api_addresses=[],
                 api_timeout=30000,
                 net_ordering_evaluation_mode=2,
                 droute_end_iter=-1,
                 verbose=1,
                 openroad_task_list=None):
        super().__init__()
        self.setDaemon(True)

        self.task_id = time.strftime("%y%m%d_%H%M%S", time.localtime(time.time())) + f'_{random.randint(0, 1000)}'
        self.task_mode = task_mode
        self.task_timeout = task_timeout
        self.testcase_name = testcase_name
        self.custom_strategies = custom_strategies
        self.parallel_workers = parallel_workers
        self.api_addresses = api_addresses
        self.api_timeout = api_timeout
        self.net_ordering_evaluation_mode = net_ordering_evaluation_mode
        self.droute_end_iter = droute_end_iter
        self.verbose = verbose
        self.openroad_task_list = openroad_task_list

        self.testcase_path = None
        self.process = None

        self.prepare()

    @property
    def task_full_name(self):
        return f'{self.testcase_name}_{self.task_mode}_{self.task_id}'

    @property
    def tcl_template_name(self):
        template_map = {
            'normal': 'normal',
            'training': 'training',
            'evaluation': 'evaluation',
        }
        return f'run.{template_map[self.task_mode]}.tcl'

    @property
    def tcl_file_name(self):
        return f'{self.testcase_name}.{self.tcl_template_name}'

    @property
    def result_path(self):
        return os.path.join(self.testcase_path, f'result_{self.task_mode}_{self.task_id}')

    def prepare(self):
        try:
            self.testcase_path = os.path.join(os.sep, 'app', 'testcases', self.testcase_name)

            if not os.path.exists(self.testcase_path):
                raise FileNotFoundError(f'{self.testcase_path} not found, please check your testcase name.')

            input_lef = os.path.join(self.testcase_path, f'{self.testcase_name}.input.lef')
            input_def = os.path.join(self.testcase_path, f'{self.testcase_name}.input.def')
            input_guides = os.path.join(self.testcase_path, f'{self.testcase_name}.input.guide')

            if not os.path.exists(self.result_path):
                os.makedirs(self.result_path, exist_ok=True)

            output_drc = os.path.join(self.result_path, f'{self.testcase_name}.output.drc.rpt')
            output_def = os.path.join(self.result_path, f'{self.testcase_name}.output.def')

            # TODO: Support multiple custom strategies, ref to src/drt/src/dr/FlexDR.cpp:strategy()
            custom_size, custom_offset = None, None
            if len(self.custom_strategies) == 1:
                custom_size, custom_offset, *_ = self.custom_strategies[0]

            template = jinja_env.get_template(self.tcl_template_name)
            script = template.render(thread_count=multiprocessing.cpu_count(),
                                     input_lef=input_lef,
                                     input_def=input_def,
                                     input_guides=input_guides,
                                     output_drc=output_drc,
                                     output_def=output_def,
                                     custom_strategies=bool(custom_size and custom_offset),
                                     custom_size=custom_size,
                                     custom_offset=custom_offset,
                                     parallel_workers=self.parallel_workers,
                                     api_address=self.api_addresses[0] if len(self.api_addresses) == 1 else None,
                                     api_timeout=self.api_timeout,
                                     net_ordering_evaluation_mode=self.net_ordering_evaluation_mode,
                                     droute_end_iter=self.droute_end_iter,
                                     verbose=self.verbose)

            with open(os.path.join(self.result_path, self.tcl_file_name), 'w') as f:
                f.write(script)

        except Exception as e:
            raise Exception(f'Failed to generate script for {self.task_full_name}, reason: {repr(e)}')

    def run(self):
        self.openroad_task_list[self.task_full_name] = self

        tcl_path = os.path.join(self.result_path, self.tcl_file_name)
        log_path = os.path.join(self.result_path, f'{self.testcase_name}.output.log')

        with open(log_path, 'w') as f:
            self.process = subprocess.Popen(['openroad', '-exit', tcl_path],
                                            stdout=f,
                                            stderr=f)

            self.process.wait(self.task_timeout)

            if self.process.returncode == 0:
                print(f'Task [{self.task_full_name}] finished successfully.')
            else:
                print(f'Task [{self.task_full_name}] failed.')

            del self.openroad_task_list[self.task_full_name]

    def terminate(self):
        self.process.terminate()
