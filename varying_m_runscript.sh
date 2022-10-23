sbatch long_main_runscript.sh 100 epsilon_greedy False s1 True 0.1 1000 4
sbatch long_main_runscript.sh 100 epsilon_greedy False s1 True 0.1 10000 4
sbatch long_main_runscript.sh 100 epsilon_greedy False s1 False 0.1 1000 4
sbatch long_main_runscript.sh 100 epsilon_greedy False s1 False 0.1 10000 4

sleep 30

sbatch long_main_runscript.sh 100 epsilon_greedy False s2 True 0.1 1000 4
sbatch long_main_runscript.sh 100 epsilon_greedy False s2 True 0.1 10000 4
sbatch long_main_runscript.sh 100 epsilon_greedy False s2 False 0.1 1000 4
sbatch long_main_runscript.sh 100 epsilon_greedy False s2 False 0.1 10000 4

sleep 30

sbatch long_main_runscript.sh 100 epsilon_greedy False s3 True 0.1 1000 4
sbatch long_main_runscript.sh 100 epsilon_greedy False s3 True 0.1 10000 4
sbatch long_main_runscript.sh 100 epsilon_greedy False s3 False 0.1 1000 4
sbatch long_main_runscript.sh 100 epsilon_greedy False s3 False 0.1 10000 4

sleep 30


sbatch long_main_runscript.sh 100 elinucb False s1 True 0. 1000 4
sbatch long_main_runscript.sh 100 elinucb False s1 True 0. 10000 4
sbatch long_main_runscript.sh 100 elinucb False s1 False 0. 1000 4
sbatch long_main_runscript.sh 100 elinucb False s1 False 0. 10000 4

sleep 30

sbatch long_main_runscript.sh 100 epsilon_greedy False s1s True 0.1 1000 1

sbatch long_main_runscript.sh 100 epsilon_greedy False s1s True 0.1 10000 1

sbatch long_main_runscript.sh 100 epsilon_greedy False s1s False 0.1 1000 1

sbatch long_main_runscript.sh 100 epsilon_greedy False s1s False 0.1 10000 1

sleep 30

sbatch long_main_runscript.sh 100 epsilon_greedy False s2s True 0.1 1000 1

sbatch long_main_runscript.sh 100 epsilon_greedy False s2s True 0.1 10000 1

sbatch long_main_runscript.sh 100 epsilon_greedy False s2s False 0.1 1000 1

sbatch long_main_runscript.sh 100 epsilon_greedy False s2s False 0.1 10000 1

sleep 30

sbatch long_main_runscript.sh 100 epsilon_greedy False s3s True 0.1 1000 1

sbatch long_main_runscript.sh 100 epsilon_greedy False s3s True 0.1 10000 1

sbatch long_main_runscript.sh 100 epsilon_greedy False s3s False 0.1 1000 1

sbatch long_main_runscript.sh 100 epsilon_greedy False s3s False 0.1 10000 1

sleep 30

sbatch long_main_runscript.sh 100 epsilon_greedy False us True 0.1 1000 1

sbatch long_main_runscript.sh 100 epsilon_greedy False us True 0.1 10000 1

sbatch long_main_runscript.sh 100 epsilon_greedy False us False 0.1 1000 1

sbatch long_main_runscript.sh 100 epsilon_greedy False us False 0.1 10000 1

sleep 30

sbatch long_main_runscript.sh 100 ucb False rus True 0. 1000 1

sbatch long_main_runscript.sh 100 ucb False rus True 0. 10000 1

sbatch long_main_runscript.sh 100 ucb False rus False 0. 1000 1

sbatch long_main_runscript.sh 100 ucb False rus False 0. 10000 1

sleep 30

sbatch long_main_runscript.sh 100 ucb False c True 0. 1000 1

sbatch long_main_runscript.sh 100 ucb False c True 0. 10000 1

sbatch long_main_runscript.sh 100 ucb False c False 0. 1000 1

sbatch long_main_runscript.sh 100 ucb False c False 0. 10000 1

sleep 30

sbatch long_main_runscript.sh 100 ucb False s1s True 0. 1000 1

sbatch long_main_runscript.sh 100 ucb False s1s True 0. 10000 1

sbatch long_main_runscript.sh 100 ucb False s1s False 0. 1000 1

sbatch long_main_runscript.sh 100 ucb False s1s False 0. 10000 1

