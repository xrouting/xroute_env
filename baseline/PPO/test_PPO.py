
import sys,pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1])) 
import time
import torch
import zmq

from baseline_utils import handle_messange
from build_3Dgrid import build_3Dgrid
import openroad_api.proto.net_ordering_pb2 as net_ordering
from PPO import PPO

#简单记录推断了多少个动作
num_action = 1
#################################### Testing ###################################
def test(checkpoint_path,port='6665',device=None):
    assert device in ['cpu','0','1']
    if device == 'cpu':
        device = torch.device(device)
    else:
        device = torch.device('cuda:'+device)

    #超参数设定
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic
    state_dim = 64

    #创建模型并加载预训练参数
    ppo_agent = PPO(state_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,device)

    print(f"loading network from : {checkpoint_path}")
    pretrained_dict=torch.load(checkpoint_path)
    model_dict=ppo_agent.policy.state_dict()
    #filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    ppo_agent.policy.load_state_dict(model_dict)
    ppo_agent.to_cpu()

    #启动监听
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:"+port)
    print('Server started.')

    while True:
        torch.cuda.empty_cache()
        ppo_agent.to_device()
        message_raw = socket.recv()

        #解析消息
        message = net_ordering.Message()
        message.ParseFromString(message_raw)
        data = handle_messange(message,socket)

        #把初始环境构建成3D网格
        routed_nets = set()
        res = build_3Dgrid(data,routed_nets,bool_inference=True)
        observation = res[0]
        action = ppo_agent.inference_action(observation)
        message = net_ordering.Message() 

        global num_action
        print(f'第 {num_action} 个 action:{action}')
        num_action += 1

        #发送网络编号
        action = int(action)
        #action为-1时直接发送，此为空布局，无需选择网络
        #其他情况下，action减去1然后再发送
        if action == -1:
            message.response.net_index = action
        else:
            message.response.net_index = int(action) - 1   
        socket.send(message.SerializeToString())    

        #清理显存
        ppo_agent.to_cpu()
        del res
        time.sleep(1)
        torch.cuda.empty_cache() 
        time.sleep(1)


if __name__ == '__main__':
    device = sys.argv[1]
    port = sys.argv[2]
    try:
        pretrained_path = sys.argv[3]
    except Exception as e:
        pretrained_path = 'results/2023-04-27--05-00-38/PPO_routing_random.pth'   
    print(f'port:{port},device:{device}')
    test(checkpoint_path=pretrained_path,port=port,device=device)
