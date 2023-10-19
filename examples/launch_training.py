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

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../baseline/openroad_api"))
sys.path.append(parent_dir)

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://*:6667')

process = None

executable_path = os.path.abspath(os.path.join(os.getcwd(), '../third_party/openroad'))
shell_script_path = os.path.abspath(os.path.join(os.getcwd(), f'../ispd/ispd18_test1/run-net-ordering-training.tcl'))
dump_workers_path = os.path.abspath(os.path.join(os.getcwd(), f'../ispd/ispd18_test1/dump'))
dir_items = os.listdir(dump_workers_path)
worker_dir_names = [name for name in dir_items if name.startswith("workerx")]

max_route_count = 10    # 所有 clip 要走几轮
route_count = 0         # 当前轮次
worker_index = 0        # 当前 clip


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

    with fileinput.FileInput(shell_script_path, inplace=True, backup='.bak') as file:
        for line in file:
            if 'worker_dir' in line:
                line = re.sub(r'workerx\d+_y\d+', worker_dir_names[worker_index], line)
            print(line, end='')

    print(f'\033[32mRouting Clip {worker_dir_names[worker_index]} For {route_count} Time(s).\033[0m')


def launch_openroad():
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

    # 等待一段时间再次启动程序
    time.sleep(1)


def main():
    while True:
        print("\033[31m%s\033[0m" % "Waiting Request...")
        socket.recv()
        socket.send(b'\0')
        print("\033[31m%s\033[0m" % "Starting OpenROAD...")

        if process:
            print("\033[31m%s\033[0m" % "Killing OpenROAD...")
            process.kill()

        t = threading.Thread(target=launch_openroad)
        t.daemon = True
        t.start()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # 判断文件是否存在
        if os.path.exists(f'{shell_script_path}.bak'):
            os.replace(f'{shell_script_path}.bak', shell_script_path)
