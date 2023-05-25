
import datetime
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1])) 
import torch
from torch.utils.tensorboard import SummaryWriter

from baseline_utils import Game
from DQN import DQN, ReplayBuffer

results_path = pathlib.Path(__file__).resolve().parents[0] / "results" / datetime.datetime.now().strftime(
    "%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
writer = SummaryWriter(results_path)

#超参数
capacity = 1000  # 经验池容量
lr = 2e-3  # 学习率
gamma = 0.99  # 折扣因子
epsilon = 0.9  # 贪心系数
target_update = 300  # 目标网络的参数的更新频率
batch_size = 8
min_size = 300  # 采样多少样本后开始训练
max_training_episodes = 4600#训练时遍历多少个Gcell
save_model_freq = 50 

# 加载环境
game = Game()

#创建经验池
replay_buffer = ReplayBuffer(capacity)

env_name = "routing"
checkpoint_path = results_path / "DQN_{}.pth".format(env_name)
print(f"save checkpoint path : {checkpoint_path}")

class Trainer:
    def __init__(self,max_training_episodes,checkpoint_path=checkpoint_path,pretrained_path=None,device=None,writer=writer):
        self.max_training_episodes = max_training_episodes
        self.checkpoint_path = checkpoint_path
        self.dqn_agent = DQN(learning_rate=lr,gamma=gamma,epsilon=epsilon,target_update=target_update,device=device)

        if pretrained_path:
            pretrained_dict=torch.load(pretrained_path)
            model_dict=self.dqn_agent.q_net.state_dict()
            #filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            #overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)         
            self.dqn_agent.q_net.load_state_dict(model_dict)
            self.dqn_agent.q_net_target.load_state_dict(model_dict)

        self.writer = writer
        self.device = device

    def save(self,checkpoint_path=None):
        if not checkpoint_path:
            checkpoint_path = self.checkpoint_path
        print(f"saving model at : {checkpoint_path}")        
        self.dqn_agent.save(checkpoint_path)
        print("model saved")

    def train(self):        
        print('store initial model')
        self.save() 

        start_time = datetime.datetime.now().replace(microsecond=0)
        print("Started training at: ", start_time)

        #num_episode决定何时停止
        num_episode = 1
        #time_step决定何时进行存储模型
        time_step = 1        
        # training loop
        while num_episode <= self.max_training_episodes:    
            assert num_episode >= 1
            print(f'num_episode={num_episode},max_training_episodes={self.max_training_episodes}')                
            # 每个回合开始前重置环境
            state,reset_try_time = game.reset()
            print('--------------')
            print(f'type(state):{type(state)}')
            num_episode += reset_try_time
            done = False            
            sum_reward_episode = 0
            sum_violation_episode = 0
            sum_via_episode = 0
            sum_wirelength_episode = 0

            while not done:
                if time_step % save_model_freq == 0:
                    self.save()      

                action = self.dqn_agent.take_action(state)

                #执行动作
                next_state,done,violation,wirelength,via = game.step(action)
                reward = -1
                reward *= violation * 500 + via * 4 + wirelength * 0.5

                sum_reward_episode += reward
                sum_violation_episode += violation
                sum_via_episode += via
                sum_wirelength_episode += wirelength    

                # 添加经验池          
                replay_buffer.add(state, action, reward, next_state, done)
                # 更新当前状态
                state = next_state
                # 找到目标就结束
                if done: 
                    break

            # 当经验池超过一定数量后，训练网络
            if replay_buffer.size() > min_size:
                # 从经验池中随机抽样作为训练集
                s, a, r, ns, d = replay_buffer.sample(batch_size)
                # 构造训练集
                transition_dict = {
                    'states': s,
                    'actions': a,
                    'next_states': ns,
                    'rewards': r,
                    'dones': d,
                }
                # 网络更新
                self.dqn_agent.update(transition_dict,self.writer)
                
            time_step += 1
            print()
            
            writer.add_scalar(
                "1.Episode/0.num_episode",
                num_episode,
                num_episode
            )             
            writer.add_scalar(
                "1.Episode/1.reward",
                sum_reward_episode,
                num_episode
            )    
            writer.add_scalar(
                "1.Episode/2.violation",
                sum_violation_episode,
                num_episode
            )  
            writer.add_scalar(
                "1.Episode/3.wirelength",
                sum_wirelength_episode,
                num_episode
            )  
            writer.add_scalar(
                "1.Episode/4.via",
                sum_via_episode,
                num_episode
            )   
        
        #存储模型
        self.save()
        # print total training time
        print("============================================================================================")
        end_time = datetime.datetime.now().replace(microsecond=0)
        print("Started training at ", start_time)
        print("Finished training at : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print("============================================================================================")

if __name__ == '__main__':
    device = sys.argv[1]
    try:
        pretrained_path = sys.argv[2]
    except IndexError as e:
        pretrained_path = None
    assert device in ['cpu','0','1','2','3']
    if device == 'cpu':
        device = torch.device(device)
    else:
        device = torch.device('cuda:'+device)    
    print(f'device={device}')
    trainer = Trainer(max_training_episodes,pretrained_path=pretrained_path,device=device)
    try:
        trainer.train()
    except KeyboardInterrupt as e:
        trainer.save()        