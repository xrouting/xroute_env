for i in range(6642,6654):
    port = str(i)
    print('nohup python test_PPO.py 0 '+ port +' > log'+port+'.txt 2>&1 &')
for i in range(6654,6666):
    port = str(i)
    print('nohup python test_PPO.py 1 '+ port +' > log'+port+'.txt 2>&1 &')
