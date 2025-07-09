#!/bin/bash

# No arguments needed for this script
# It runs all hardcoded (algo_name, style, epsilon) combinations

possibilities=(
    "epsilon_greedy u 1."
    "epsilon_greedy u 0.1"
    "epsilon_greedy i 0.1"
    "epsilon_greedy r 0.1"
    "epsilon_greedy c 0.1"
    "ucb u 0."
    "ucb i 0."
)

# Loop over all possibilities and call the script
for entry in "${possibilities[@]}"; do
    read -r algo_name style epsilon <<< "$entry"
    ./share_interval_main_driver_runscript.sh "$algo_name" "$style" "$epsilon"
done

