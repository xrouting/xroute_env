# XRoute Environment
XRoute Environment, standing for self-learning (denoted by X) for detailed routing (denoted by Route), is a novel reinforcement learning environment to train agents to order and route all nets in various challenging testcases efficiently and effectively, and present the routing results in varying dashboards.

## Quickstart

### Installation

To interact with the xroute environment, you need to download the simulator first:

| Operating System | Download Link |
| --- | --- |
| Ubuntu 22.04 | [Download](https://github.com/xplanlab/OpenROAD/releases/tag/v0.0.1) |

Then, put the simulator in the `third_party/openroad` folder.

You may also need to execute the following command to install some libraries to ensure that OpenRoad can start up properly.

```bash
cd third_party/openroad
chmod +x DependencyInstaller.sh
source ./DependencyInstaller.sh
```

### Launch Agent

[DQN](./baseline/DQN/README.md)

[PPO](./baseline/DQN/PPO.md)

### Launch Simulator

Run the following command to get launch script:

```bash
cd examples
python3 launch.py [start_port][worker_num]
```

start_port: the listen port number of the first worker instance.

worker_num: the number of worker instances.

### TODO List

- [ ] Auto download ispd testcases
- [ ] Support distributed routing on one server

### Acknowledgement

The routing simulator in xroute environment is mainly based on [OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD) TritonRoute. Thanks for their wonderful work!

