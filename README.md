# xroute_env
RL environment for detailed routing.

## Quickstart

### Installation

To interact with the xroute environment, you need to download the simulator first:

| Operating System | Download Link |
| --- | --- |
| Ubuntu 22.04 | [Download](https://drive.google.com/file/d/1-Zxd0HiOHclNtwCON5wOM78eCzsPrOBB/view?usp=sharing) |

Then, put the simulator in the `third_party/openroad` folder.

### Launch Algorithm Backend

[DQN](./baseline/DQN/README.md)

[PPO](./baseline/DQN/PPO.md)

### Launch Simulator

Run the following command to get launch script:

```bash
cd examples
python3 init.py [start_port][worker_num]
```

start_port: the listen port number of the first worker instance.

worker_num: the number of worker instances.

### TODO List

- [ ] Auto download ispd testcases
- [ ] Support distributed routing on one server

### Acknowledgement

The routing simulator in xroute environment is mainly based on [OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD) TritonRoute. Thanks for their wonderful work!

### Citations

Please cite the paper and star this repo if you use xroute environment or find it interesting/useful, thanks! Feel free to open an issue if you have any questions.

```
@misc{zhou2023xroute,
      title={XRoute Environment: A Novel Reinforcement Learning Environment for Routing}, 
      author={Zhanwen Zhou and Hankz Hankui Zhuo and Xiaowu Zhang and Qiyuan Deng},
      year={2023},
      eprint={2305.13823},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
