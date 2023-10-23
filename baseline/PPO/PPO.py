import torch
import math
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from baseline_utils import normalize,RepresentationNetwork,mlp


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

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

class ActorCritic(nn.Module):
    def __init__(self, state_dim,device):
        super(ActorCritic, self).__init__()
        self.device = device
        self.representation_network = RepresentationNetwork(device)
        self.actor = Actor(device)

        # critic
        self.critic =  mlp(
                state_dim,
                layer_sizes=[64,64],
                output_size=1,
                output_activation=torch.nn.Identity,
                activation=torch.nn.Tanh,
            )

    def representation(self, observation):
        '''
        args:
            observation:numpy.array.[N,不定长的channel,D,H,W]
        res:
            encoded_state:[-1,self.rnn_hidden_dim]
            action_mapping_list:list[dict[key:int,value:Tensor]].每个value维度是[self.rnn_hidden_dim]
        '''
        encoded_state,action_mapping_list = self.representation_network(observation)
        encoded_state = normalize(encoded_state)
        if action_mapping_list:
            for i in range(len(action_mapping_list)):
                for action,state in action_mapping_list[i].items():
                    action_mapping_list[i][action] = normalize([state])[0]
        return encoded_state,action_mapping_list        

    def forward(self):
        raise NotImplementedError
    
    def get_policy_from(self,encoded_state,action_mapping_list):
        '''遍历action_mapping_list中的每个a,计算p(s,a)
        args:
            encoded_state:Tensor.[N,hidden_size]
            action_mapping_list:list[dict[key:int,value:Tensor]].每个value维度是[self.rnn_hidden_dim]
        return:
            policy_list:list.形如[ [(netId,prob),...] ]
        '''
        policy_list = []#[N,不定长list]
        if len(action_mapping_list) > 0:
            #episode loop
            for i in range(len(action_mapping_list)):
                policy = {}
                envRepsentation= encoded_state[i].unsqueeze(0)#[1,hidden_size]
                #net loop
                netOrder = []
                logits = []        
                for netId,netState in action_mapping_list[i].items():
                    #注意要以batch形式输入
                    logit = self.actor(envRepsentation,netState.unsqueeze(0))
                    netOrder.append(netId)
                    logits.append(logit)
                logits = torch.Tensor(logits)
                logits = torch.softmax(logits,dim=0)

                for j in range(len(logits)):
                    tmp_logits = logits[j]
                    policy[netOrder[j]] = tmp_logits
                policy = sorted(policy.items(),key=lambda x:x[1],reverse=True)
                policy_list.append(policy)         
        else:
            print(f'inf get_policy_from function,len(action_mapping_list) is 0')
            print(f'action_mapping_list:{action_mapping_list}')
        return policy_list 

    def act(self, state):
        '''
        仅用于单个game playing,即batch_size为1
        '''
        encoded_state,action_mapping_list = self.representation(state)
        policy_list = self.get_policy_from(encoded_state,action_mapping_list)
        if len(policy_list) == 1:
            policy_list = policy_list[0]
            policy_dict = dict(policy_list)
            actionIds = list(policy_dict.keys())
            probs = torch.Tensor(list(policy_dict.values()))
            dist = Categorical(probs)
            
            #采样动作
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            action = actionIds[action]
            action = torch.as_tensor(action)
            state_val = self.critic(encoded_state.to(self.device))

            return action, action_logprob, state_val
    
    def evaluate(self, state, action):
        '''
        args:
            action:list.[Tensor]      
        '''
        action_logprobs = []
        state_values = []
        dist_entropy = []
        for i in range(len(state)):
            encoded_state,action_mapping_list = self.representation(state[i])
            policy_list = self.get_policy_from(encoded_state,action_mapping_list)
            if len(policy_list) == 1:
                policy_list = policy_list[0]            
            policy_dict = dict(policy_list)
            actionIds = list(policy_dict.keys())
            probs = torch.Tensor(list(policy_dict.values()))
            tmp_action = action[i]

            dist = Categorical(probs.to(tmp_action.device))
            if action[i].device == torch.device('cpu'):
                tmp_action = tmp_action.cpu()

            action_logprobs.append(torch.log(policy_dict[tmp_action.item()]))
            dist_entropy.append(dist.entropy())
            state_values.append(self.critic(encoded_state).squeeze(0).squeeze(0))
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device=None):
        self.device = device

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, self.device)
        self.policy.to(device)

        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, self.device)
        self.policy_old.to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def to_device(self,bool_inference=True):
        self.policy.to(self.device)
        if not bool_inference:
            self.policy_old.to(self.device)

    def to_cpu(self):
        device = torch.device('cpu')
        self.policy.to(device)
        self.policy_old.to(device)

    def select_action(self, state):        
        '''
        仅用于单个game playing,即batch_size为1
        '''        
        if state.device == torch.device('cpu'):
            self.buffer.states.append(state.numpy().tolist())
        else:
            self.buffer.states.append(state.cpu().numpy().tolist())
        action, action_logprob, _ = self.policy_old.act(torch.FloatTensor(state).to(self.device))                    
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state)
        return action.item()
        
    def inference_action(self,state):
        self.policy.eval()
        encoded_state,action_mapping_list = self.policy.representation(state)
        if len(action_mapping_list) > 0:
            policy_list = self.policy.get_policy_from(encoded_state,action_mapping_list)     
            print(f'policy_list:{policy_list}')
            try:
                return policy_list[0][0][0]   
            except Exception as e:
                print(e)
                print('send aciton -1!!!')
                return -1
        return None
    
    def create_batch_and_cal_loss(self,training_fragment_len,batch_size):
        loss = 0
        for _ in range(batch_size):
            #build mini-batch traning data
            rewards = []
            discounted_reward = 0
            ep_max_len = len(self.buffer.rewards)
            start_index = np.random.choice(ep_max_len-training_fragment_len)

            for reward, is_terminal in zip(reversed(self.buffer.rewards[start_index:start_index+training_fragment_len]), \
                                           reversed(self.buffer.is_terminals[start_index:start_index+training_fragment_len])):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            #Normalizing the rewards
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

            old_states = np.array(self.buffer.states[start_index:start_index+training_fragment_len])
            old_actions = torch.squeeze(torch.stack(self.buffer.actions[start_index:start_index+training_fragment_len], dim=0)).to(self.device)
            old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs[start_index:start_index+training_fragment_len], dim=0)).to(self.device)

            # calculate advantages
            advantages = []
            for i in range(len(rewards)):
                encoded_state,_ = self.policy.representation(self.buffer.state_values[start_index:start_index+training_fragment_len][i])
                state_val = self.policy.critic(encoded_state)
                advantages.append(rewards[i] - state_val.squeeze(0))
            advantages = torch.Tensor(advantages).to(self.device)

            try:
                len(old_actions)
            except TypeError:
                old_actions = [old_actions]
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            logprobs = torch.Tensor(logprobs).to(self.device)
            state_values = torch.Tensor(state_values).to(self.device)
            dist_entropy = torch.Tensor(dist_entropy).to(self.device)
            
            #ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)

            #Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss += -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
        return loss.mean()

    def update(self,training_fragment_len,batch_size):
        # Optimize policy for K epochs
        loss_list = []
        for i in range(self.K_epochs):
            #build mini-batch traning data
            loss = self.create_batch_and_cal_loss(training_fragment_len,batch_size)
            loss_list.append(loss)
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()            
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        print(f'loss_list:{loss_list}')
        print('upadating model finished')
        return loss_list
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       


