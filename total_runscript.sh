sleep 3600
echo "epsilon_greedy False s2 True 0.1 10000 4"
./long_main_driver_runscript.sh epsilon_greedy False s2 True 0.1 10000 4
sleep 1800
echo "epsilon_greedy True s3 True 0.1 10000 4"
./long_main_driver_runscript.sh epsilon_greedy True s3 True 0.1 10000 4
sleep 1800
echo "epsilon_greedy False s3 True 0.1 10000 4"
./long_main_driver_runscript.sh epsilon_greedy False s3 True 0.1 10000 4