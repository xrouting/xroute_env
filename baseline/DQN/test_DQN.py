
import sys,pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1])) 
import time
import torch
import zmq

from baseline_utils import handle_messange
from build_3Dgrid import build_3Dgrid
from DQN import DQN
import openroad_api.proto.net_ordering_pb2 as net_ordering


#简单记录推断了多少个动作
num_action = 1

def test(checkpoint_path,port=None,device=None):
    assert device in ['cpu','0','1']
    if device == 'cpu':
        device = torch.device(device)
    else:
        device = torch.device('cuda:'+device)

    lr = 2e-3  # 学习率
    gamma = 0.99  # 折扣因子
    epsilon = 0.9  # 贪心系数
    target_update = 300  # 目标网络的参数的更新频率

    #创建模型并载入预训练参数
    dqn_agent = DQN(learning_rate=lr,gamma=gamma,epsilon=epsilon,target_update=target_update,device=device)
    print(f"loading network from : {checkpoint_path}")
    pretrained_dict=torch.load(checkpoint_path)
    model_dict=dqn_agent.q_net.state_dict()
    #filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    dqn_agent.q_net.load_state_dict(model_dict)

    dqn_agent.to_cpu()
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:" + port)
    print('Server started.')

    while True:
        torch.cuda.empty_cache()
        dqn_agent.to_device()
        message_raw = socket.recv()

        #解析消息
        message = net_ordering.Message()
        message.ParseFromString(message_raw)
        data = handle_messange(message,socket)
        import json
        json_str = json.dumps(data)
        with open('./test.json','w',encoding='utf-8') as f:
            f.write(json_str)

        #把初始环境构建成3D网格
        routed_nets = set()
        res = build_3Dgrid(data,routed_nets,bool_inference=True)
        observation = res[0]

        #调用模型推断action
        action = dqn_agent.inference_action(observation)
        message = net_ordering.Message() 
        
        #发送action
        global num_action
        print(f'第 {num_action} 个 action:{action}')
        num_action += 1
        action = int(action)
        #action为-1时直接发送，此为空布局，无需选择网络
        #其他情况下，action减去1然后再发送
        if action == -1:
            message.response.net_index = action
        else:
            message.response.net_index = int(action) - 1          
        socket.send(message.SerializeToString())    

        #清理显存
        dqn_agent.to_cpu()
        del res
        time.sleep(1)
        torch.cuda.empty_cache() 
        time.sleep(1)

        


if __name__ == '__main__':
    port = sys.argv[1]
    device = sys.argv[2]
    print(f'port:{port},device:{device}')
    test(checkpoint_path='results/2023-05-10--11-06-26/DQN_routing.pth',port=port,device=device)
