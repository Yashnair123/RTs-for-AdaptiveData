for weight in "True" "False"; do
    for T in {10..100..10}; do
        sbatch interval_main_runscript.sh $T $1 $2 $weight $3 100 $4
        sleep 0.5
    done
done
