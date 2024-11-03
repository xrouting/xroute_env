import torch
import zmq
from utils import handle_messange,Game
import openroad_api.proto.net_ordering_pb2 as net_ordering
import time
import json
from discrete_A3C import Net


class InferenceSelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, checkpoint_path=None,port='6666',device='cpu'):
        #self.model.set_weights(initial_checkpoint["weights"])
        #加载模型
        self.model = Net()
        pretrained_dict=torch.load(checkpoint_path)
        model_dict=self.model.state_dict()
        #filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)          
        self.game = Game()
        #assert device in ['cpu','0','1']
        if device == 'cpu':
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda:'+device)        
        self.model.to(self.device)
        self.model.eval()
        self.port = port
        self.idx = 1

    def response_action(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:"+self.port)
        print('Server started.')
        while True:
            message_raw = socket.recv()
            #socket.send(b'world')
            #解析消息
            message = net_ordering.Message()
            message.ParseFromString(message_raw)
            data = handle_messange(message,socket)
            #observation = get_feature(data)
            observation = self.game.get_feature(data)
            action_list,_ = self.model.choose_action(observation,bool_inference=True)
            action_list = [int(action) - 1 for action in action_list]
            message = net_ordering.Message() 
            message.response.net_list.extend(action_list)
            print(f'第 {self.idx} 次 发送 action_list = {action_list}')
            self.idx += 1
            socket.send(message.SerializeToString())             


if __name__ == '__main__':
    import sys
    port = sys.argv[1]
    device = sys.argv[2]
    try:
        pretrained_path = sys.argv[3]
    except Exception as e:
        pretrained_path = '/home/dengqiyuan/XRoute/baseline/A3C/episode_train_8000/2023-07-14--10-18-25/A3C_routing.pth'
    print(f'port:{port},device:{device}')
    server = InferenceSelfPlay(checkpoint_path=pretrained_path,port=port,device=device)
    server.response_action()  