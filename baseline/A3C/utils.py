"""
Functions that use multiple times
"""

from torch import nn
import torch
import numpy as np


def v_wrap(np_array, dtype=np.float32):
    '''把numpy向量转成Tensor
    '''
    if type(np_array) is list:
        np_array = np.array(np_array)
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    elif type(np_array) is np.float32:
        np_array = np.array([np_array])
    else:
        pass
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(opt, lnet, gnet, done, s_, buffer_s, buffer_a, buffer_r, gamma):
    '''
    args:
        s_:当前状态
        buffer_s:历史状态列表
        buffer_a:历史动作列表
        buffer_r:历史奖赏列表
        buffer_done
    '''
    #if done:
    #    v_s_ = 0.               # terminal
    #else:
    #    v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    #buffer_v_target = []
    #for r in br[::-1]:    # reverse buffer r
    #    v_s_ = r + gamma * v_s_
    #    buffer_v_target.append(v_s_)
    #buffer_v_target.reverse()

    loss = lnet.loss_func(
        s_,
        buffer_s,
        #v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        buffer_a,
        buffer_r,
        done,
        gamma)

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
        
    osd = opt.state_dict()
    for _, bufs in osd["state"].items():
        if "step" in bufs.keys():
            # convert state_step back from int into a singleton tensor
            bufs["step"] = torch.tensor(bufs["step"])        
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )

import zmq
import openroad_api.proto.net_ordering_pb2 as net_ordering
import time
import json

def handle_messange(message,socket):
    data = None
    if message.HasField('request'):
        req = message.request
        #print(f'req.count_map:{req.count_map}')
        if req.count_map:
            try:
                count_map = {str(int(key)+1): value for key, value in json.loads(req.count_map).items()}
            except Exception as e:
                print(f'req.count_map:{req.count_map}')
                import pickle
                path = '/home/dengqiyuan/XRoute/baseline/A3C/count_map.pkl'
                pickle.dump({'count_map':req.count_map}, open(path, "wb"))
        else:
            count_map = {}
        if req.metrics_delta:
            metrics_delta = {str(int(key)+1): value for key, value in json.loads(req.metrics_delta).items()}
        else:
            metrics_delta = {}
                
        data = [
            [req.dim_x, req.dim_y, req.dim_z],
            [],
            [req.reward_violation, req.reward_wire_length, req.reward_via],
            list(map(lambda x: x + 1, req.nets)),  # 在 XRoute 中，net 是从 1 开始的
            req.openroad,
            req.xroute,    
            count_map,    
            metrics_delta    
        ]

        for node in req.nodes:
            node_type = 0
            if node.type == net_ordering.ACCESS:
                node_type = node.net + 1  # 在 XRoute 中，net 是从 1 开始的
            elif node.type == net_ordering.BLOCKAGE:
                node_type = -1

            node_pin = -1
            if node.type == net_ordering.NodeType.ACCESS:
                node_pin = node.pin + 1  # 在 XRoute 中，pin 是从 1 开始的

            info = [
                [node.maze_x, node.maze_y, node.maze_z],
                [node.point_x, node.point_y, node.point_z],
                [int(node.is_used), node_type, node_pin],
            ]
            data[1].append(info)        
        if req.is_done:
            socket.send(b'\0')
           
    return data

def get_netSet(data):
    obstacles = []
    accessPoints = {} #dict[netid:{pinid:[...]}]
    for vertex in data[1]:
        bool_occupy,Net,Pin = vertex[2]
        if Net == -1:
            obstacles.append(vertex)
        elif Net == 0:
            #普通点被占用，视同obstacle
            if bool_occupy == 1:
                obstacles.append(vertex)
        else:
            #vertex是pin点的情况
            assert type(Net) is int and Net >= 1
            #pin点被占用，既要添加到obstacle，也仍然保留在accessPoints中
            if bool_occupy == 1:
                obstacles.append(vertex)            
            if Net in accessPoints:
                if Pin not in accessPoints[Net]:
                    accessPoints[Net][Pin] = [vertex]
                else:
                    accessPoints[Net][Pin].append(vertex)
            else:
                accessPoints[Net] = {}
                accessPoints[Net][Pin] = [vertex]
    return list(accessPoints.keys())    

class Game:
    def __init__(self,seed=None,server_ip='127.0.0.1',server_port='6666',client_ip='127.0.0.1',client_port='5555'):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_ip = client_ip
        self.client_port = client_port

        self.context = None
        self.req_socket = None
        self.rep_socket = None

        self.data = None
        self.observation = None
        self.accessPoints = None

    def to_play(self):
        return 0
    
    def _cal_reward(self,cost_array):
        violation,wirelength,via = cost_array
        return 0.5*wirelength + 4*via + 500*violation

    def _getAccessPoints(self,data):
        accessPoints = {} #dict[netid:{pinid:[...]}]
        for vertex in data[1]:
            _,Net,_ = vertex[2]
            #vertex = vertex[1]
            if Net not in [-1,0]:
                #vertex是pin点的情况
                assert type(Net) is int and Net >= 1         
                if Net in accessPoints:
                    accessPoints[Net].append(vertex)
                else:
                    accessPoints[Net] = [vertex]     
        return accessPoints   

    def get_feature(self,data):
        '''从后端传送过来的数据中抽取22个特征
        '''
        observation = {}
        count_map = data[6]
        metrics_delta = data[7]
        accessPoints = {}
        #if not self.accessPoints:
        #    self.accessPoints = self._getAccessPoints(data)
        #accessPoints = self.accessPoints
        if not self.accessPoints:
            self.accessPoints = self._getAccessPoints(data)
            accessPoints= self.accessPoints
        else:
            accessPoints = self.accessPoints
        #print(f'receive netlist:{sorted(list(accessPoints.keys()))}')

        for net in accessPoints.keys():
            feats = []
            
            if self.observation:
                HPL = self.observation[net][0]
                conflict = self.observation[net][1]
                LA = self.observation[net][2:18]                

            else:
                accessPoint = accessPoints[net]
                x_list,y_list,z_list,layer_list = [],[],[],[]
                for point in accessPoint:
                    x_list.append(point[1][0])
                    y_list.append(point[1][1])
                    z_list.append(point[1][2])
                    layer_list.append(point[0][2])
                #计算半周长
                HPL = max(x_list) - min(x_list) + max(y_list) - min(y_list) + max(z_list) - min(z_list)          
                      
                #计算重叠
                conflict = 0
                for _net in accessPoints.keys():
                    if _net == net:
                        pass
                    for ap in accessPoints[_net]:
                        if min(x_list) <= ap[1][0] <= max(x_list) and min(y_list) <= ap[1][1] <= max(y_list) and min(z_list) <= ap[1][2] <= max(z_list):
                            conflict += 1
                            break
                
                #遍历net的pin点数据来计算Layer Assignment
                layer_set = set(layer_list)
                LA = [0] * 16
                for layer in layer_set:
                    LA[layer] = 1

            ##静态特征
            feats.append(HPL)
            feats.append(conflict)
            feats.extend(LA)       

            ##动态特征
            #获取count
            feats.append(count_map.get(str(net),0))    
            #获取violation,wirelength,via
            feats.extend(metrics_delta.get(str(net),[0,0,0]))     
            observation[net] = np.array(feats)              

        self.observation = observation
        return observation
    
    def send_0(self):
        if not self.rep_socket:
            context = zmq.Context()
            self.rep_socket = context.socket(zmq.REP)
            self.rep_socket.bind('tcp://'+self.server_ip+':'+self.server_port)           
        self.rep_socket.send(b'\0')

    def step(self,action_list,total_step):
        """
        Apply action to the game.

        Args:
            action_list : list.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """        
        done = False
        observation = None
        #发送action
        
        if not self.rep_socket:
            context = zmq.Context()
            self.rep_socket = context.socket(zmq.REP)
            self.rep_socket.bind('tcp://'+self.server_ip+':'+self.server_port)          
        message = net_ordering.Message() 
        action_list = [int(a)-1 for a in action_list]
        message.response.net_list.extend(action_list)
        self.rep_socket.send(message.SerializeToString()) 
        print(f'total_step:{total_step},send action_list:{action_list}')

        #接收新的环境信息并解析
        message_raw = self.rep_socket.recv()
        
        # REP 在 recv 后还要 send 一下作为结束
        #self.rep_socket.send(b'\0')

        # 解析消息
        message = net_ordering.Message()
        message.ParseFromString(message_raw)
        data = handle_messange(message,self.rep_socket)
        print(f'receive data:{len(data)}')

        openroad_cost = data[4]
        xroute_cost = data[5]
        try:
            reward = self._cal_reward(openroad_cost) - self._cal_reward(xroute_cost)
        except Exception as e:
            reward = 0
        print(f'receive cost of xroute:{xroute_cost}，receive cost of openroad:{openroad_cost}')

        #mismath penalty
        alpha = 0.1 if total_step <= 100 else 0
        k = len(action_list)
        penalty = 0
        for i in range(len(action_list)):
            penalty += (action_list[i] - i) ** 2
        reward -= alpha / k * penalty

        #先看是否done
        if len(xroute_cost) > 0:
            xroute_violation = xroute_cost[0]
            if xroute_violation == 0:
                done = True
        
        #整理得到22个特征
        next_observation = self.get_feature(data)     
        return reward,done,next_observation

    def reset(self,bool_jump=False,bool_reset=False):
        """Reset the game for a new game.
        Returns:
            observation:Dict.键是int，表示某个net，值是numpy.array，对应这个net的22个特征，
        """
        ###向后台发送初始化请求
        notDone = True
        print('reset env...')
        #observation = []
        while notDone:
            if not self.req_socket:
                context = zmq.Context()
                self.req_socket = context.socket(zmq.REQ)
                #self.req_socket.connect(f'tcp://127.0.0.1:5555') 
                self.req_socket.connect('tcp://'+self.client_ip+':'+self.client_port)
                self.rep_socket = context.socket(zmq.REP)
                #self.rep_socket.bind(f'tcp://127.0.0.1:6666')      
                self.rep_socket.bind('tcp://'+self.server_ip+':'+self.server_port)      
            #通知后台初始化GCell  
            if bool_jump:
                #跳转到下一个布局
                self.req_socket.send(b'jump')
                self.data = None
                self.observation = None
                self.accessPoints = None
                print(f'send `jump`')
            elif bool_reset:
                #把布局重新跳回第一个
                self.req_socket.send(b'reset')
                print(f'send `reset`')
            else:
                #初始化当前布局
                self.req_socket.send(b'initial')

            # REQ 在 send 后还要 recv 一下作为结束
            self.req_socket.recv()

            #接收后台的初始环境
            message_raw = self.rep_socket.recv()

            #解析消息
            message = net_ordering.Message()
            message.ParseFromString(message_raw)
            self.data = handle_messange(message,self.rep_socket)
            #print(len(self.data))

            #若reset到空布局则跳过
            action_space = get_netSet(self.data)
            if len(action_space) != 0:
                notDone = False     
                #整理得到22个特征
                self.observation = self._get_feature(self.data)                       
            else:
                #self.rep_socket原本需要在step函数里面被调用来返回netlist，对于空布局只能返回b'\0'
                self.rep_socket.send(b'\0')                  
                print(f'跳过空布局,len(action_space)={len(action_space)}')

        print('reset env done')
        return self.observation

    def close(self):
        return None

if __name__ == '__main__':
    game = Game()
    game.reset()