set OR $argv0
set worker0 [exec $OR worker0.tcl > logs/worker0.log &]
set worker1 [exec $OR worker1.tcl > logs/worker1.log &]
set worker2 [exec $OR worker2.tcl > logs/worker2.log &]
set worker3 [exec $OR worker3.tcl > logs/worker3.log &]
set worker4 [exec $OR worker4.tcl > logs/worker4.log &]
set worker5 [exec $OR worker5.tcl > logs/worker5.log &]
set worker6 [exec $OR worker6.tcl > logs/worker6.log &]
set worker7 [exec $OR worker7.tcl > logs/worker7.log &]
set worker8 [exec $OR worker8.tcl > logs/worker8.log &]
set worker9 [exec $OR worker9.tcl > logs/worker9.log &]
set worker10 [exec $OR worker10.tcl > logs/worker10.log &]
set worker11 [exec $OR worker11.tcl > logs/worker11.log &]
set worker12 [exec $OR worker12.tcl > logs/worker12.log &]
set worker13 [exec $OR worker13.tcl > logs/worker13.log &]
set worker14 [exec $OR worker14.tcl > logs/worker14.log &]
set worker15 [exec $OR worker15.tcl > logs/worker15.log &]

set balancer [exec $OR balancer.tcl > logs/balancer.log &]

read_lef /home/plan/ispd/tests/ispd18_test1/ispd18_test1.input.lef
read_def /home/plan/ispd/tests/ispd18_test1/ispd18_test1.input.def
read_guides /home/plan/ispd/tests/ispd18_test1/ispd18_test1.input.guide

set_thread_count 32
detailed_route_debug -parallel_workers 16
detailed_route -verbose 1 \
               -distributed \
               -remote_host 127.0.0.1 \
               -remote_port 9998 \
               -cloud_size 16 \
               -shared_volume results

exec kill $worker0
exec kill $worker1
exec kill $worker2
exec kill $worker3
exec kill $worker4
exec kill $worker5
exec kill $worker6
exec kill $worker7
exec kill $worker8
exec kill $worker9
exec kill $worker10
exec kill $worker11
exec kill $worker12
exec kill $worker13
exec kill $worker14
exec kill $worker15

exec kill $balancer
