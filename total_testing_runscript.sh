echo "100 epsilon_greedy False u True 1. 1000 4"
sbatch short_main_runscript.sh 100 epsilon_greedy False u True 1. 1000 4
sbatch short_main_runscript.sh 100 epsilon_greedy False u True 1. 10000 4
sbatch short_main_runscript.sh 100 epsilon_greedy False u False 1. 1000 4
sbatch short_main_runscript.sh 100 epsilon_greedy False u False 1. 10000 4

sleep 5

echo "100 epsilon_greedy False u True 0.1 1000 4"
sbatch short_main_runscript.sh 100 epsilon_greedy False u True 0.1 1000 4
sbatch short_main_runscript.sh 100 epsilon_greedy False u True 0.1 10000 4
sbatch short_main_runscript.sh 100 epsilon_greedy False u False 0.1 1000 4
sbatch short_main_runscript.sh 100 epsilon_greedy False u False 0.1 10000 4

echo "100 epsilon_greedy False s1 True 0.1 1000 4"
sbatch short_main_runscript.sh 100 epsilon_greedy False s1 True 0.1 1000 4
sbatch short_main_runscript.sh 100 epsilon_greedy False s1 True 0.1 10000 4
sbatch short_main_runscript.sh 100 epsilon_greedy False s1 False 0.1 1000 4
sbatch short_main_runscript.sh 100 epsilon_greedy False s1 False 0.1 10000 4

sleep 5

echo "100 epsilon_greedy False s2 True 0.1 1000 4"
sbatch short_main_runscript.sh 100 epsilon_greedy False s2 True 0.1 1000 4
sbatch short_main_runscript.sh 100 epsilon_greedy False s2 True 0.1 10000 4
sbatch short_main_runscript.sh 100 epsilon_greedy False s2 False 0.1 1000 4
sbatch short_main_runscript.sh 100 epsilon_greedy False s2 False 0.1 10000 4

sleep 5

echo "100 epsilon_greedy False s3 True 0.1 1000 4"
sbatch short_main_runscript.sh 100 epsilon_greedy False s3 True 0.1 1000 4
sbatch short_main_runscript.sh 100 epsilon_greedy False s3 True 0.1 10000 4
sbatch short_main_runscript.sh 100 epsilon_greedy False s3 False 0.1 1000 4
sbatch short_main_runscript.sh 100 epsilon_greedy False s3 False 0.1 10000 4

sleep 5

echo "100 elinucb False u True 0. 1000 4"
sbatch short_main_runscript.sh 100 elinucb False u True 0. 1000 4
sbatch short_main_runscript.sh 100 elinucb False u True 0. 10000 4
sbatch short_main_runscript.sh 100 elinucb False u False 0. 1000 4
sbatch short_main_runscript.sh 100 elinucb False u False 0. 10000 4

sleep 5

echo "100 elinucb False s1 True 0. 1000 4"
sbatch short_main_runscript.sh 100 elinucb False s1 True 0. 1000 4
sbatch short_main_runscript.sh 100 elinucb False s1 True 0. 10000 4
sbatch short_main_runscript.sh 100 elinucb False s1 False 0. 1000 4
sbatch short_main_runscript.sh 100 elinucb False s1 False 0. 10000 4