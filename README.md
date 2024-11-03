# XRoute Environment
XRoute Environment, standing for self-learning (denoted by X) for detailed routing (denoted by Route), is a novel reinforcement learning environment to train agents to order and route all nets in various challenging testcases efficiently and effectively, and present the routing results in varying dashboards.

## Quickstart

### Installation

To interact with the xroute environment, you need to download the simulator first:

| Operating System | Download Link |
| --- | --- |
| Ubuntu 22.04 | [Download](https://drive.google.com/file/d/1tgyXDrM5VqEHoo_SFlUhy5d8vAQdh5SX/view?usp=drive_link) |

Then, put the simulator in the `third_party/openroad` folder.

You may also need to execute the following command to install some libraries to ensure that OpenRoad can start up properly.

```bash
cd third_party/openroad
chmod +x DependencyInstaller.sh
source ./DependencyInstaller.sh
```

### Agent Introduction

[DQN](./baseline/DQN/README.md)

[PPO](./baseline/PPO/README.md)

[A3C](./baseline/A3C/README.md)

[MCTS](./baseline/xroute/README.md)

### Launch Mode

You can choose to launch the simulator in following modes:

#### Training Mode

In this mode, the simulator should launch first, then the agent can control the simulator to train the model.(The usage of xroute can be found in the README.md under the xroute directory.)

```bash
cd examples && python3 launch_training.py

cd baseline/DQN && python3 train_DQN.py cpu
# cd baseline/PPO && python3 train_PPO.py cpu
# cd baseline/A3C && python discrete_A3C.py
```

After executing the command above, the simulator will listen to the port 6667 to wait for environment reset command, and then interact with the agent via port 5556.

#### Evaluation Mode

In this mode, the agent should launch first, then the simulator can connect to the agent to get the action.

```bash
cd baseline/DQN && python3 test_DQN.py cpu 5556
# cd baseline/PPO && python3 test_PPO.py cpu 5556
# cd baseline/A3C && python3 test_A3C.py 5556 cpu

cd examples && python3 launch_evaluation.py 5556
```

### Acknowledgement

The routing simulator in xroute environment is mainly based on [OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD) TritonRoute. Thanks for their wonderful work!
