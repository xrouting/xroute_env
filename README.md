# xroute_env
RL environment for detailed routing.

## Quickstart

### Installation

To interact with the xroute environment, you need to download the simulator first:

| Operating System | Download Link |
| --- | --- |
| Ubuntu 22.04 | [Download](https://drive.google.com/file/d/1-Zxd0HiOHclNtwCON5wOM78eCzsPrOBB/view?usp=sharing) |

Then, put the simulator in the `third_party/openroad` folder.

Before launching this executable, you need to install some dependency libraries. Execute the following command to do so.

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

