set_thread_count 40

detailed_route_debug \
        -skip_reroute \
        -parallel_workers 1 \
        -api_addr 127.0.0.1:25565 \
        -api_timeout 86400000 \
        -net_ordering_training 2 \
        -graph_mode 1

# Need to modify if the dumped regions are not at iteration 0
detailed_route_worker_debug \
        -maze_end_iter 3 \
        -drc_cost 8 \
        -marker_cost 0 \
        -follow_guide 1 \
        -ripup_mode 1

detailed_route_run_worker \
        -dump_dir /home/plan/zhanwen/eda/DumpWorkers/ispd18_test1/7x7 \
        -worker_dir workerx359100_y239400