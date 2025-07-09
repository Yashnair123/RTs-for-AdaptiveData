for num_samples in 10 100; do
    for T in {10..100..10}; do
        sbatch share_interval_main_runscript.sh $T $1 $2 $3 $num_samples
        sleep 0.5
    done
done
