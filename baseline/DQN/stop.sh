ps aux | grep test_DQN.py | awk {'print$2'} | xargs kill -9
