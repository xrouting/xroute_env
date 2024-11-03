ps aux | grep discrete | awk {'print$2'} | xargs kill -9
