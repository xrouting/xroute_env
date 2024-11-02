# USAGE

### 1.train

#### Server-side Startup

Set the path correctly in trainer4/main.py

````
cd	xroute_env/baseline/xroute/trainer4
./launcher.sh
````

#### Client-side Startup

​	1.Set the parameters in `net_order.py`, especially line 24, you need to choose training mode (`train`) 

2. Run 

```
python xroute.py
```

Follow the on-screen prompts that the script provides

### 2.infer

#### Server-side Startup

```
cd xroute_env/baseline/xroute/xr-11fea-ispd18test1
```

Set the path correctly in `init.py`.

```
python init.py Port number_of_threads
cd scripts && ./launcher.sh
```

#### Client-side Startup

​	1.Set the parameters in `net_order.py`, especially line 24, you need to choose  mode (`inference_step_by_step`).

2. Run 

```
python xroute.py
```

Follow the on-screen prompts that the script provides

