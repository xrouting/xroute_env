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

### Agent Introduction

[DQN](./baseline/DQN/README.md)

[PPO](./baseline/DQN/PPO.md)

### Launch Mode

You can choose to launch the simulator in following modes:

#### Training Mode

In this mode, the simulator should launch first, then the agent can control the simulator to train the model.

```bash
cd examples && python3 launch_training.py

cd baseline/DQN && python3 train_DQN.py cpu
# cd baseline/PPO && python3 train_PPO.py cpu
```

After executing the command above, the simulator will listen to the port 6667 to wait for environment reset command, and then interact with the agent via port 5556.

#### Inference Mode

In this mode, the agent should launch first, then the simulator can connect to the agent to get the action.

```bash
cd baseline/DQN && python3 test_DQN.py cpu 5556
# cd baseline/PPO && python3 test_PPO.py cpu 5556

cd examples && python3 launch_inference.py 5556
```

### TODO List

- [ ] Auto download ispd testcases
- [ ] Support distributed routing on one server

### Acknowledgement

The routing simulator in xroute environment is mainly based on [OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD) TritonRoute. Thanks for their wonderful work!
