for T in {10..100..10}; do
    sbatch short_main_runscript.sh $T $1 $2 $3 $4 $5 $6 $7
    sleep 0.5
done