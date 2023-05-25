follow https://www.nature.com/articles/nature14236


#1.train DQN
python train_DQN.py GPU_id [pretrained_path]    #The GPU_id value should be one of ['cpu','0','1','2','3']

#2.test DQN
#Method 1
python test_DQN.py GPU_id PORT [pretrained_path]


#method 2
or you can modify inference.py and execute 
python inference.py > inference.sh
bash inference.sh

and you can execute  `bash start_reboot.sh` to reboot inference server when it crash because of torch.cuda.OutOfMemoryError
