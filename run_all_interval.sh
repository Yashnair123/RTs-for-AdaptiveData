#!/bin/bash

# Check if a type argument is given
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <type>"
    echo "Type must be either 'confidence' or 'conformal'"
    exit 1
fi

type=$1

case $type in
    confidence)
        possibilities=(
            "epsilon_greedy uu_X 1."
            "epsilon_greedy ui_X 0.1"
            "epsilon_greedy rui_X 0.1"
            "epsilon_greedy comb 0.1"
            "ucb ui_X 0."
            "ucb rui_X 0."
            "ucb comb 0."
        )
        ;;
    conformal)
        possibilities=(
            "epsilon_greedy u 1."
            "epsilon_greedy u 0.1"
            "epsilon_greedy i 0.1"
            "epsilon_greedy r 0.1"
            "epsilon_greedy c 0.1"
            "ucb u 0."
            "ucb i 0."
        )
        ;;
    *)
        echo "Invalid type: $type"
        echo "Type must be either 'confidence' or 'conformal'"
        exit 1
        ;;
esac

# Loop over all possibilities and call the script
for entry in "${possibilities[@]}"; do
    read -r algo_name style epsilon <<< "$entry"
    ./interval_main_driver_runscript.sh "$algo_name" "$style" "$epsilon" "$type"
done

