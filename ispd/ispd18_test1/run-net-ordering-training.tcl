set_debug_level DRT autotuner 1
detailed_route_debug -dr -api_host 192.168.0.100:5556 -net_ordering_use_api -net_ordering_train -api_timeout 600000
detailed_route_worker_debug -maze_end_iter 3 -drc_cost 8 -marker_cost 0 -follow_guide 1 -ripup_mode 1

set current_directory [file normalize [file dirname [info script]]]
set dump_folder_path [file normalize [file join $current_directory "../../ispd/ispd18_test1/dump"]]
puts $dump_folder_path

detailed_route_run_worker -dump_dir $dump_folder_path \
                          -worker_dir workerx39900_y319200
