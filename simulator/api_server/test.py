from utils.openroad import create_task_by_testcase_name
from threading import Thread


import signal
import sys


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    Thread(target=create_task_by_testcase_name, args=('ispd18_test1',), kwargs=({"verbose": 1})).start()

