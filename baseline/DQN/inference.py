
for i in range(6654,6660):
    port = str(i)
    print('nohup python test_DQN.py '+ port +' 0 > log'+port+'.txt &')

for i in range(6660,6666):
    port = str(i)
    print('nohup python test_DQN.py '+ port +' 1 > log'+port+'.txt &')    
