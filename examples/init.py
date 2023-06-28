import os
import shutil
import sys

folder_path = 'scripts'

# 删除 scripts 目录
if os.path.exists(folder_path):
    shutil.rmtree(folder_path)

# 初始化目录结构
os.makedirs(folder_path, exist_ok=True)
os.makedirs(folder_path + '/logs', exist_ok=True)
os.makedirs(folder_path + '/results', exist_ok=True)


# 生成脚本

worker_template = '''set_thread_count 1
detailed_route_debug -custom_strategies -custom_size 1 -custom_offset 0 -net_ordering_use_api -api_host 127.0.0.1:{api_port}
run_worker -host 127.0.0.1 -port {worker_port}
'''
balancer_template = 'add_worker_address -host 127.0.0.1 -port {worker_port}'
leader_launch_template = 'set worker{i} [exec $OR worker{i}.tcl > logs/worker{i}.log &]'
leader_terminate_template = 'exec kill $worker{i}'

api_port_start = int(sys.argv[1])
worker_count = int(sys.argv[2])
worker_port_start = 8800

balancer_content = ''
leader_launch_content = ''
leader_terminate_content = ''

for i in range(worker_count):
    api_port = api_port_start + i
    worker_port = worker_port_start + i

    worker_content = worker_template.format(api_port=api_port, worker_port=worker_port)
    file_name = f'worker{i}.tcl'

    balancer_content += balancer_template.format(worker_port=worker_port) + '\n'
    leader_launch_content += leader_launch_template.format(i=i) + '\n'
    leader_terminate_content += leader_terminate_template.format(i=i) + '\n'

    with open(f'{folder_path}/{file_name}', 'w') as f:
        f.write(worker_content)


balancer_content += f'run_load_balancer -host 127.0.0.1 -port {worker_port_start - 1}' + '\n'
with open(f'{folder_path}/balancer.tcl', 'w') as f:
    f.write(balancer_content)


# 生成 leader 脚本
leader_content = f'''set OR $argv0
{leader_launch_content}
set balancer [exec $OR balancer.tcl > logs/balancer.log &]

set parent_dir [file dirname [file dirname [pwd]]]
set ispd18_test1_path [file join $parent_dir ispd/ispd18_test1]

read_lef ${{ispd18_test1_path}}/ispd18_test1.input.lef
read_def ${{ispd18_test1_path}}/ispd18_test1.input.def
read_guides ${{ispd18_test1_path}}/ispd18_test1.input.guide

set_thread_count 40

detailed_route_debug -custom_strategies -custom_size 1 -custom_offset 0
detailed_route -output_maze ${{ispd18_test1_path}}/ispd18_test1.output.maze.log \\
               -output_drc ${{ispd18_test1_path}}/ispd18_test1.output.drc.rpt \\
               -verbose 1 \\
               -distributed \\
               -remote_host 127.0.0.1 \\
               -remote_port {worker_port_start - 1} \\
               -cloud_size {worker_count} \\
               -shared_volume results

{leader_terminate_content}
write_def ${{ispd18_test1_path}}/ispd18_test1.output.def
'''

with open(f'{folder_path}/leader.tcl', 'w') as f:
    f.write(leader_content)

launcher_content = 'rm -rf logs/* && ../../third_party/openroad/openroad -exit leader.tcl | tee logs/leader.log'

launcher_path = f'{folder_path}/launcher.sh'
with open(launcher_path, 'w') as f:
    f.write(launcher_content)

os.chmod(launcher_path, 0o755)

# 生成日志监控脚本
monitor_content = '''while true; do
    for file in logs/worker*.log; do
        echo -n "[$file] "
        tail -n 1 "$file"
    done
    echo ""
    echo ""
    sleep 10
done
'''
monitor_path = f'{folder_path}/monitor.sh'
with open(monitor_path, 'w') as f:
    f.write(monitor_content)

os.chmod(monitor_path, 0o755)
