for null in "True" "False"; do
    for weight in "True" "False"; do
        ./main_driver_runscript.sh $1 $null $2 $weight $3 100 $4
    done
done
