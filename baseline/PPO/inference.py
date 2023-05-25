for i in range(6642,6654):
    port = str(i)
    print('nohup python test_PPO.py '+ port +' 0 > log'+port+'.txt 2>&1 &')
for i in range(6654,6666):
    port = str(i)
    print('nohup python test_PPO.py '+ port +' 1 > log'+port+'.txt 2>&1 &')
