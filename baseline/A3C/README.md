The baseline A3C algorithm follows the original paper in https://ieeexplore.ieee.org/document/9557780

### 1.train A3C

Server-side Startup

````
cd	xroute_env/baseline/A3C/openroad_api/demo/xroute_simulation_platform
python trainer_auto_switch_2.py
````

Client-side Startup

```
cd xroute_env/baseline/A3C
./start_train.sh
```

### 2.test A3C


Client-side Startup


````
python test_A3C.py PORT device [pretrained_path]
````

