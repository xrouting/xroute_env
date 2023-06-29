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
                 task_mode='evaluation',
                 task_timeout=86400 * 365,
                 testcase_name='ispd18_sample',
                 droute_end_iter=-1,
                 verbose=1):
        super().__init__()
        self.setDaemon(True)

        self.task_id = time.strftime("%y%m%d_%H%M%S", time.localtime(time.time())) + f'_{random.randint(0, 1000)}'
        self.task_mode = task_mode
        self.task_timeout = task_timeout
        self.testcase_name = testcase_name
        self.droute_end_iter = droute_end_iter
        self.verbose = verbose

        self.testcase_path = None
        self.process = None

    @property
    def task_full_name(self):
        return f'{self.testcase_name}_{self.task_mode}_{self.task_id}'

    @property
    def tcl_template_name(self):
        template_map = {
            'evaluation': 'normal',
        }
        return f'run.{template_map[self.task_mode]}.tcl'

    @property
    def tcl_file_name(self):
        return f'{self.testcase_name}.{self.tcl_template_name}'

    @property
    def result_path(self):
        return os.path.join(self.testcase_path, f'result_{self.task_mode}_{self.task_id}')

    def generate_script(self):
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

        template = jinja_env.get_template(self.tcl_template_name)
        script = template.render(thread_count=multiprocessing.cpu_count(),
                                 input_lef=input_lef,
                                 input_def=input_def,
                                 input_guides=input_guides,
                                 output_drc=output_drc,
                                 output_def=output_def,
                                 droute_end_iter=self.droute_end_iter,
                                 verbose=self.verbose)

        with open(os.path.join(self.result_path, self.tcl_file_name), 'w') as f:
            f.write(script)

    def run(self):
        self.generate_script()

        tcl_path = os.path.join(self.result_path, self.tcl_file_name)
        log_path = os.path.join(self.result_path, f'{self.testcase_name}.output.log')

        with open(log_path, 'w') as f:
            self.process = subprocess.Popen(['openroad', '-exit', tcl_path],
                                       stdout=f,
                                       stderr=f)

            self.process.wait(self.task_timeout)

            # TODO: handle different type of error
            if self.process.returncode == 0:
                print(f'Task [{self.task_full_name}] finished successfully.')
            else:
                print(f'Task [{self.task_full_name}] failed.')

    def terminate(self):
        self.process.terminate()
