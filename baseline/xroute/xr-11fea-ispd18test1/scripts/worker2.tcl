set_thread_count 1
detailed_route_debug -parallel_workers 1 -skip_reroute -api_addr 127.0.0.1:10890 -api_timeout 1800000 -net_ordering_evaluation 1 -graph_mode 1
run_worker -host 127.0.0.1 -port 10002
