The baseline PPO algorithm follows the original paper in https://arxiv.org/pdf/1707.06347.pdf

#1.train PPO
python train_PPO.py GPU_id [pretrained_path]    #The GPU_id value should be one of ['cpu','0','1','2','3']

model checkpoint path:baseline/PPO/results

----Communication setting
see ../baseline_utils.py class Game

--reset method
socket = context.socket(zmq.REQ)
socket.connect('tcp://127.0.0.1:'+self.port_initial) #port_initial default 6667
#inform backend to initialize environment
socket.send(b'initial') 

#create new socket to receive initial environment
self.socket = context.socket(zmq.REP)
self.socket.bind('tcp://*:'+self.port_recv) #port_recv default 5556
message_raw = self.socket.recv()


--step method
self.socket = context.socket(zmq.REP)
self.socket.bind('tcp://*:'+ self.port_recv)  #same as above self.socket

#send action to backend
self.socket.send(message.SerializeToString())
#receive new environment information
message_raw = self.socket.recv()

#default model checkpoint path is PPO/results

#2.test PPO
#Method 1
python test_PPO.py GPU_id PORT [pretrained_path]


#method 2
#or you can modify inference.py and execute 
python inference.py > inference.sh
bash inference.sh

#and you can execute  `bash start_reboot.sh` to reboot inference server when it crash because of torch.cuda.OutOfMemoryError
