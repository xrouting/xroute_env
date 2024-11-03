# coding=utf-8
import os
import select
import subprocess
import sys
import threading
import time

import zmq

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(parent_dir)

process = None


def launch_openroad():
    # 启动程序
    global process

    # 程序位置
    executable_path = os.path.abspath(os.path.join(os.getcwd(), '../../../../cmake-build-release/src'))

    process = subprocess.Popen([f'{executable_path}/openroad', '-exit',
                                '/home/plan/eda/OpenROAD/src/drt/test/results/ispd18_test1/run-net-ordering-train.tcl'],
                                # '/Users/matts8023/Home/Career/SYSU/eda/OpenROAD_Local/src/drt/test/results/ispd18_test1/run-debug-worker.tcl'],
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


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://*:6666')

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
