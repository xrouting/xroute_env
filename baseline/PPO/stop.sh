#!/bin/bash
ps aux | grep test_PPO | awk {'print$2'} | xargs kill -9
