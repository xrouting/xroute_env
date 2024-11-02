# coding=utf-8
import fileinput
import os
import re
import select
import subprocess
import sys
import threading
import time

import zmq

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(parent_dir)

import proto.net_ordering_pb2 as net_ordering

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://*:5555')

process = None
thread_exit = False

testcase_name = 'ispd18_test1'
clip_size = 7

executable_path = os.path.abspath(os.path.join(os.getcwd(), '../../../../cmake-build-release/src'))           # OpenROAD 可执行文件路径
shell_script_path1 = os.path.abspath(os.path.join(os.getcwd(), f'../../../../src/drt/test/results/{testcase_name}/run-net-ordering-training.tcl'))
shell_script_path2 = os.path.abspath(os.path.join(os.getcwd(), f'../../../../src/drt/test/results/{testcase_name}/run-net-ordering-training-xr.tcl'))

dump_workers_path = f'/home/plan/eda/DumpWorkers/{testcase_name}/{clip_size}x{clip_size}'

dir_items = os.listdir(dump_workers_path)
worker_dir_names = [name for name in dir_items if name.startswith("workerx")]

max_route_count = 1000000000 # 单个 region 循环次数
route_count = 0 # 当前 region 循环次数
worker_index = 0


def modify_script():
    global route_count
    global worker_index

    if route_count == max_route_count:
        route_count = 1

        if worker_index == len(worker_dir_names) - 1:
            worker_index = 0
            print(f'\033[33mNew Turning.\033[0m')
        else:
            worker_index += 1
    else:
        route_count += 1

    with fileinput.FileInput(shell_script_path1, inplace=True, backup='.bak') as file:
        for line in file:
            if 'dump_dir' in line:
                line = re.sub(r'(dump_dir\s)(.+)(\s\\)$', f'\\1{dump_workers_path}\\3', line)
            if 'worker_dir' in line:
                line = re.sub(r'workerx\d+_y\d+', worker_dir_names[worker_index], line)
            print(line, end='')

    with fileinput.FileInput(shell_script_path2, inplace=True, backup='.bak') as file:
        for line in file:
            if 'dump_dir' in line:
                line = re.sub(r'(dump_dir\s)(.+)(\s\\)$', f'\\1{dump_workers_path}\\3', line)
            if 'worker_dir' in line:
                line = re.sub(r'workerx\d+_y\d+', worker_dir_names[worker_index], line)
            print(line, end='')

    print(f'\033[32mUse Testcase {testcase_name} In {testcase_name}/{clip_size}x{clip_size} With Worker {worker_dir_names[worker_index]} For {route_count} Time(s).\033[0m')


def launch_openroad(shell_script_path):
    global process

    modify_script()

    process = subprocess.Popen([f'{executable_path}/openroad', '-exit', shell_script_path],
                               cwd=executable_path,
                               stdout=subprocess.PIPE)

    # 创建一个 select 对象，将程序的输出文件描述符添加到其中
    select_obj = select.poll()
    select_obj.register(process.stdout, select.POLLIN)

    while True:
        # 等待文件描述符就绪
        ready = select_obj.poll(1000)  # 等待 1000 毫秒

        if ready:
            # 读取程序的输出
            output = process.stdout.readline().decode().strip()

            # 处理程序的输出
            print(output)

        # 检查程序是否已经返回
        if process.poll() is not None:
            break


def run_task():
    global thread_exit

    try:
        os.remove(f'{executable_path}/training2.txt')
    except FileNotFoundError:
        pass

    launch_openroad(shell_script_path1)

    if thread_exit:
        return

    launch_openroad(shell_script_path2)

    if thread_exit:
        return

    with open(f'{executable_path}/training2.txt', 'r') as f:
        lines = f.readlines()

        message = net_ordering.Message()
        result = message.request
        result.openroad.extend([int(x) for x in lines[0].split()])
        result.xroute.extend(list(map(int, lines[1].split())))
        result.count_map = lines[2]
        result.metrics_delta = lines[3]

        _socket = context.socket(zmq.REQ)
        _socket.connect('tcp://localhost:6666')
        _socket.send(message.SerializeToString())

        print(lines)
        print('Send result successfully.')


def main():
    global thread_exit
    global worker_index

    while True:
        print("\033[31m%s\033[0m" % "Waiting Request...")
        msg = socket.recv()

        # 手动切换布局
        if msg == b'reset':
            print("\033[31m%s\033[0m" % "Reset...")
            worker_index = 0
        elif msg == b'jump':
            print("\033[31m%s\033[0m" % "Jump...")
            if worker_index == len(worker_dir_names) - 1:
                worker_index = 0
                print(f'\033[33mNew Turning.\033[0m')
            else:
                worker_index += 1

        socket.send(b'\0')
        print("\033[31m%s\033[0m" % "Starting OpenROAD...")

        thread_exit = True
        if process:
            print("\033[31m%s\033[0m" % "Killing OpenROAD...")
            process.kill()
        time.sleep(1)
        thread_exit = False

        t = threading.Thread(target=run_task)
        t.daemon = True
        t.start()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # 判断文件是否存在
        if os.path.exists(f'{shell_script_path1}.bak'):
            os.replace(f'{shell_script_path1}.bak', shell_script_path1)
        if os.path.exists(f'{shell_script_path2}.bak'):
            os.replace(f'{shell_script_path2}.bak', shell_script_path2)
