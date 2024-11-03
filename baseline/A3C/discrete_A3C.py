"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import torch
import torch.nn as nn
from utils import v_wrap, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_optimizer import SharedOptimizer
from utils import Game
import numpy as np
import pathlib
import os
import datetime

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.99
#MAX_EP = 3000
MAX_EP = 1600


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    #logger.info(f"&&&sizes:{sizes}")
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)

class Net(nn.Module):
    def __init__(self, law_dim=22, s_dim=64, a_dim=1):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.encoder = mlp(input_size=law_dim,
                           layer_sizes=[64],
                           output_size=s_dim,
                           output_activation=torch.nn.Tanh,
                           activation=torch.nn.Tanh)
        
        #策略网络输出正态部分的均值和方差
        self.policy_net = mlp(input_size=s_dim,
                           layer_sizes=[],
                           output_size=2)

        #价值网络
        self.value_net = mlp(input_size=s_dim,
                           layer_sizes=[],
                           output_size=1)

        #set_init([self.pi1, self.pi2, self.v1, self.v2])
        #self.distribution = torch.distributions.Categorical
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        encode_vector = self.encoder(x)
        mean,variance = self.policy_net(encode_vector)
        value = self.value_net(encode_vector)
        return mean,variance,value

        #pi1 = torch.tanh(self.pi1(x))
        #logits = self.pi2(pi1)
        #v1 = torch.tanh(self.v1(x))
        #values = self.v2(v1)
        #return logits, values

    def choose_action(self, x_dict, bool_inference=False):
        '''
        输入所有net列表，逐个进行打分，按照评分得到并返回网络编号排序
        '''
        self.eval()
        
        score_dict = {}
        for k in x_dict.keys():
            mean,variance,_ = self.forward(v_wrap(x_dict[k]))
            score = None
            if bool_inference:
                score = mean.detach().numpy()
            else:
                m = self.distribution(mean, variance**2)
                score = m.sample().numpy()
            score_dict[k] = score
        score_list = sorted(score_dict.items(),key=lambda x:x[1],reverse=True)
        net_list = [item[0] for item in score_list]
        return net_list,score_dict


    def loss_func(self, s_,s_list, a_list, r_list,done,gamma):
        '''
        args:
            s_:dict.当前状态，包含所有net的22个特征
            s_list:历史状态列表,每个元素都是一个字典，包含所有net的22个特征
            a_list:历史状态列表，每个元素都是一个字典，对应所有net的得分（注意，不是概率，而是正态分布生成的得分）
            r_list:单步奖赏列表，每个元素是一个float
            done:bool,标志是否完成game
        '''
        self.train()
        assert len(s_list) == len(a_list) == len(r_list)

        #记录每个net对应的初始R,初始化为0
        R_mapping = {}
        for net in s_.keys():
            R_mapping[net] = 0
            if not done:
                m,var,value = self.forward(v_wrap(s_[net]))
                R_mapping[int(net)] = value

        #逆序遍历buffer的所有元素        
        total_loss = 0
        n_buffer = len(a_list)
        for i in range(len(a_list)-1,-1,-1):
            s = s_list[i]
            a = a_list[i]
            r = r_list[i]

            #遍历每个动作
            for net in s.keys():
                #policy loss
                m,var,value = self.forward(v_wrap(s[net]))
                R_mapping[net] = r + gamma ** (n_buffer-i) * R_mapping[int(net)]

                td = R_mapping[net] - value
                value_loss = td.pow(2)

                m = self.distribution(m,var**2)
                policy_loss = -m.log_prob(v_wrap(a[net])) * td.detach().squeeze()

                entropy = m.entropy()
                #total_loss += (value_loss + policy_loss).mean()
                total_loss += value_loss * 0.25 + policy_loss + entropy * 0.001
        return total_loss
    
    def save(self,checkpoint_path=None):
        if not checkpoint_path:
            checkpoint_path = self.checkpoint_path
        print(f"saving model at : {checkpoint_path}")        
        torch.save(self.state_dict(), checkpoint_path)
        print("model saved")

    def load(self, checkpoint_path):        
        self.load_state_dict(torch.load(checkpoint_path))
        print(f'load model from {checkpoint_path} done')


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name,server_port='6666',client_port='5555'):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net()           # local network
        self.env = Game(server_port=server_port,client_port=client_port)

    def run(self):
        total_step = 1
        #启动训练时把布局调整到第一个
        bool_reset = True
        bool_jump = False
        n_per_layout = 10
        s = {}
        while self.g_ep.value < MAX_EP:
            print(f'self.g_ep.value = {self.g_ep.value} , MAX_EP = {MAX_EP}')
            #训练交互前重置为第一个布局，每交互完n_per_layout个布局则跳转下一个布局
            #空布局一律跳过
            while s == {} or bool_jump:
                if bool_reset:
                    s = self.env.reset(bool_reset=True)
                    bool_reset = False
                else:
                    s = self.env.reset(bool_jump=True)
                if s != {} and bool_jump:
                    bool_jump = False
                    break
            
            buffer_s, buffer_a, buffer_r = [], [], []
            # = 0.

            while True:
                #print(f'total_step:{total_step}')
                net_list,score_dict = self.lnet.choose_action(s)
                #a = self.lnet.choose_action(v_wrap(s[None, :]))
                #s_, r, done, _,_ = self.env.step(net_list)
                r,done,next_s = self.env.step(net_list,total_step)
                #self.env.send_0()
                s = next_s
                print(f'r={r},done={done}')
                #if done: r = -1
                #ep_r += r
                buffer_a.append(score_dict)
                #buffer_s.append(s)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    print('into break loop')
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, next_s, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    #if done:  # done and print information
                    #    record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                    with self.g_ep.get_lock():
                        self.g_ep.value += 1   
                    total_step += 1      
                    self.env.send_0()  
                    bool_jump = True         
                    break
                
                total_step += 1

                # initial 环境
                self.env.send_0()
                self.env.reset()
                
        print(f'self.g_ep.value = {self.g_ep.value} , MAX_EP = {MAX_EP}')     
        self.res_queue.put(None)


if __name__ == "__main__":
    #print(f'N_S={N_S},N_A={N_A}')#4 2
    gnet = Net()        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedOptimizer(gnet.parameters(), lr=1e-3)      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    results_path = pathlib.Path(__file__).resolve().parents[0] / "results" / datetime.datetime.now().strftime(
    "%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
    env_name = "routing"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        print(f'create dir:{results_path}')
    checkpoint_path = results_path / "A3C_{}.pth".format(env_name)
    print(f"set checkpoint path : {checkpoint_path}")
    # parallel training
    #workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i,\
                      server_port=str(6666+i),client_port=str(5555+i)) for i in range(8)]
    #gnet.save(checkpoint_path)
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    gnet.save(checkpoint_path)

    #import matplotlib.pyplot as plt
    #plt.plot(res)
    #plt.ylabel('Moving average ep reward')
    #plt.xlabel('Step')
    #plt.show()
