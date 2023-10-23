import collections
import numpy as np
import pathlib
import random
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F

from baseline_utils import normalize,RepresentationNetwork,mlp

class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done

    def size(self):
        return len(self.buffer)
   
class Actor(torch.nn.Module):
    def __init__(
        self,
        device
    ):
        super().__init__()
        self.device = device
        self.mlp_policy = mlp(input_size=128,layer_sizes=[128,64],output_size=1)
        
    def forward(self,x,action):
        '''
        args:
            x:[N,hidden_size]
            action:[N,hidden_size]
        return:
            logit:[N,1]
        '''
        logit = self.mlp_policy(torch.concat([x,action],dim=1).to(self.device))
        return logit

class RepActor(torch.nn.Module):
    def __init__(self,device):
        super(RepActor, self).__init__()
        self.device = device
        self.representation_network = RepresentationNetwork(device)
        self.actor = Actor(device)        

class DQN:
    def __init__(self,learning_rate, gamma, epsilon,
                 target_update, device=None):
        super(DQN, self).__init__()
        self.learning_rate = learning_rate  # 训练时的学习率
        self.gamma = gamma  # 折扣因子，对下一状态的回报的缩放
        self.epsilon = epsilon  # 贪婪策略，有1-epsilon的概率探索
        self.target_update = target_update  # 目标网络的参数的更新频率
        self.device = device
        # 计数器，记录迭代次数
        self.count = 0

        #由于需要匹配可变的动作空间，原始版本的Q网络需要变成此处的self.representation_network和self.actor两部分
        self.q_net = RepActor(device)
        self.q_net.to(device)
        
        self.q_net_target = RepActor(device)
        self.q_net_target.to(device)
        self.q_net_target.load_state_dict(self.q_net.state_dict())    

        # 优化器，更新训练网络的参数
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

    def representation(self, observation):
        '''
        args:
            observation:numpy.array.[N,不定长的channel,D,H,W]
        res:
            encoded_state:[-1,self.hidden_dim]
            action_mapping_list:list[dict[key:int,value:Tensor]].每个value维度是[self.hidden_dim]
        '''
        encoded_state,action_mapping_list = self.q_net.representation_network(observation)
        encoded_state = normalize(encoded_state)
        if action_mapping_list:
            for i in range(len(action_mapping_list)):
                for action,state in action_mapping_list[i].items():
                    action_mapping_list[i][action] = normalize([state])[0]
        return encoded_state,action_mapping_list 
    
    def to_device(self,bool_inference=True):
        self.q_net.to(self.device)
        if not bool_inference:
            self.q_net_target.to(self.device)

    def to_cpu(self):
        device = torch.device('cpu')
        self.q_net.to(device)
        self.q_net_target.to(device)   

    def get_policy_from(self,encoded_state,action_mapping_list,actor_net=None,bool_prob=True):
        '''遍历action_mapping_list中的每个action,计算action的概率
        args:
            encoded_state:Tensor.[N,hidden_size]
            action_mapping_list:list[dict[key:int,value:Tensor]].每个value维度是[self.hidden_dim]
        return:
            policy_list:list.形如[ [(netId,prob),...] ]
        '''
        if actor_net is None:
            actor_net = self.q_net.actor
        policy_list = []#[N,不定长list]
        if action_mapping_list:
            #episode loop
            for i in range(len(action_mapping_list)):
                policy = {}
                envRepsentation= encoded_state[i].unsqueeze(0)#[1,hidden_size]
                #net loop
                netOrder = []
                logits = []        
                for netId,netState in action_mapping_list[i].items():
                    #注意要以batch形式输入
                    logit = actor_net(envRepsentation,netState.unsqueeze(0))
                    netOrder.append(netId)
                    logits.append(logit)
                
                logits = torch.Tensor(logits)
                if bool_prob:
                    logits = torch.softmax(logits,dim=0)
                for j in range(len(logits)):
                    tmp_logits = logits[j]
                    policy[netOrder[j]] = tmp_logits
                policy = sorted(policy.items(),key=lambda x:x[1],reverse=True)
                policy_list.append(policy) 
        return policy_list 

    def take_action(self, state):
        '''训练时动作选择
        仅用于单个game playing,即batch_size为1
        '''        
        encoded_state,action_mapping_list = self.representation(state)
        policy_list = self.get_policy_from(encoded_state,action_mapping_list)[0]

        # 如果小于该值就取最大的值对应的索引
        if np.random.random() < self.epsilon:
            action = policy_list[-1][0]
        # 如果大于该值就随机探索
        else:
            # 随机选择一个动作
            policy_dict = dict(policy_list)
            actionIds = list(policy_dict.keys())           
            action = np.random.choice(actionIds)
        print(f'action={action}')
        return action

    def inference_action(self,state):
        '''推断时动作选择'''
        self.q_net.eval()
        encoded_state,action_mapping_list = self.representation(state)
        if len(action_mapping_list) > 0:
            policy_list = self.get_policy_from(encoded_state,action_mapping_list,actor_net=self.q_net.actor)     
            print(f'policy_list:{policy_list}')
            try:
                return policy_list[0][0][0]   
            #如果给到一个没有网络的空布局，则特殊处理
            except Exception as e:
                print(e)
                print('send aciton -1!!!')
                return -1 
        return None

    def cal_loss(self,transition_dict):
        states = transition_dict['states']
        actions = torch.tensor(transition_dict['actions']).view(-1,1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1)
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)        
        next_states = transition_dict['next_states']
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1)

        total_loss = 0
        for i in range(len(states)):
            encoded_state,action_mapping_list = self.representation(states[i])
            policy_list = self.get_policy_from(encoded_state,action_mapping_list,bool_prob=False)[0]    
            policy_dict = dict(policy_list)

            #获取动作轨迹中动作对应的当前网络预测概率
            if len(policy_list) == 0:
                q_values = 0
            else:
                action = actions[i].item()
                q_values = policy_dict[action]

            #目标网络预测的最大动作的概率
            encoded_state,action_mapping_list_target = self.representation(next_states[i])
            policy_list = self.get_policy_from(encoded_state,action_mapping_list,actor_net=self.q_net_target.actor,bool_prob=False)[0]  
            if len(policy_list) == 0:
                print(f'policy_list:{policy_list}')
                print(f'action_mapping_list:{action_mapping_list}')
                print(f'action_mapping_list_target:{action_mapping_list_target}')
                max_next_q_values = 0
            else:              
                max_next_q_values = policy_list[-1][1]
            q_targets = rewards[i].cpu() + self.gamma * max_next_q_values * (1-dones[i])
            dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
            total_loss += dqn_loss
        return total_loss.mean()

    def update(self, transition_dict,writer):
        dqn_loss = self.cal_loss(transition_dict)
        writer.add_scalar(
            "2.Training/1.Loss",
            dqn_loss,
            self.count
        )  
        dqn_loss.requires_grad_(True)
        print()
        print(f'dqn_loss:{dqn_loss}')
        print()
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        #更新目标网络
        if self.count % self.target_update == 0:
            self.copy_network()
        
        self.count += 1
    
    def copy_network(self):
        print('同步目标网络')
        self.q_net_target.load_state_dict(self.q_net.state_dict())     

    def save(self, checkpoint_path):
        torch.save(self.q_net.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):        
        self.q_net.load_state_dict(torch.load(checkpoint_path))
        self.q_net_target.load_state_dict(torch.load(checkpoint_path))
        print(f'load model from {checkpoint_path} done')