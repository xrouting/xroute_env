
import datetime
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2])) 
import torch
from torch.utils.tensorboard import SummaryWriter
results_path = pathlib.Path(__file__).resolve().parents[0] / "results" / datetime.datetime.now().strftime(
    "%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
writer = SummaryWriter(results_path)

from baseline_utils import Game
from PPO import PPO

#超参数
max_training_episodes = 4600
save_model_freq = int(max_training_episodes / 10)
update_timestep = 100      # update policy every n timesteps
K_epochs = 10               # update policy for K epochs in one PPO update
training_fragment_len = 5 #训练时计算loss所用数据的步长大小
assert training_fragment_len <= update_timestep
batch_size = 8
state_dim = 64 # state space dimension
eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor
lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

game = Game()
env_name = "routing"
checkpoint_path = results_path / "PPO_{}.pth".format(env_name)

class Trainer:
    def __init__(self,max_training_episodes,pretrained_path=None,device=None):
        self.max_training_episodes = max_training_episodes
        self.checkpoint_path = checkpoint_path
        self.ppo_agent = PPO(state_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,device)
        if pretrained_path:
            pretrained_dict=torch.load(pretrained_path)
            model_dict=self.ppo_agent.policy.state_dict()
            #filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            #overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)         
            self.ppo_agent.policy.load_state_dict(model_dict)
            self.ppo_agent.policy_old.load_state_dict(model_dict)

        self.train_step = 1

    def save(self,checkpoint_path=None):
        if not checkpoint_path:
            checkpoint_path = self.checkpoint_path
        print("--------------------------------------------------------------------------------------------")
        print(f"saving model at : {checkpoint_path}")        
        self.ppo_agent.save(checkpoint_path)
        print("model saved")
         
    def train(self):
        start_time = datetime.datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)

        #num_episode决定何时停止
        num_episode = 1
        #time_step决定何时进行存储模型和更新模型
        time_step = 1
        # training loop
        while num_episode <= self.max_training_episodes:
            assert num_episode >= 1
            print(f'num_episode={num_episode},max_training_episodes={self.max_training_episodes}')
            state,reset_try_time = game.reset()
            num_episode += reset_try_time
            done = False
            sum_reward_episode = 0
            sum_violation_episode = 0
            sum_via_episode = 0
            sum_wirelength_episode = 0

            while not done:
                #定期更新模型
                if time_step % update_timestep == 0:
                    loss_list = self.ppo_agent.update(training_fragment_len,batch_size)
                    #self.ppo_agent.update(writer)
                    for i in range(len(loss_list)):
                        writer.add_scalar(
                            "2.Training/1.Loss",
                            loss_list[i],
                            self.train_step
                        )  
                        self.train_step += 1   

                #定期存储模型
                if time_step % save_model_freq == 0:
                    self.save()        

                #计算各个net的概率并依照这个概率分布采样一个动作
                action = self.ppo_agent.select_action(state)        

                #获得新的环境信息        
                state,done,violation,wirelength,via = game.step(action)
                time_step += 1
                reward = -1
                reward *= violation * 500 + via * 4 + wirelength * 0.5

                sum_reward_episode += reward
                sum_violation_episode += violation
                sum_via_episode += via
                sum_wirelength_episode += wirelength                                               

                #存储信息
                self.ppo_agent.buffer.rewards.append(reward)
                self.ppo_agent.buffer.is_terminals.append(done)

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
        
        self.save()
        print("============================================================================================")
        end_time = datetime.datetime.now().replace(microsecond=0)
        print("Started training at : ", start_time)
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

    
    
    
    
    
    
    
