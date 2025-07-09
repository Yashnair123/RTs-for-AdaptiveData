for dim in {3..50}; do
    for weight in "True" "False"; do
        sbatch main_runscript_varying_dim.sh $1 $2 $weight $3 $dim
    done
    sleep 0.5
done

