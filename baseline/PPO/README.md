follow https://arxiv.org/pdf/1707.06347.pdf

#1.train PPO
python train_DQN.py GPU_id [pretrained_path]    #The GPU_id value should be one of ['cpu','0','1','2','3']

#default model checkpoint path is PPO/results

#2.test PPO
#Method 1
python test_DQN.py GPU_id PORT [pretrained_path]


#method 2
#or you can modify inference.py and execute 
python inference.py > inference.sh
bash inference.sh

#and you can execute  `bash start_reboot.sh` to reboot inference server when it crash because of torch.cuda.OutOfMemoryError