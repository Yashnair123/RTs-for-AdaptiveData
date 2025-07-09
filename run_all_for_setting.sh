#!/bin/bash

# Check if a setting argument is given
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <setting>"
    exit 1
fi

setting=$1

case $setting in
    0)
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
    1|5)
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
    2)
        possibilities=(
            "epsilon_greedy uu_X 1."
            "epsilon_greedy i_X 0.1"
            "epsilon_greedy ui_X 0.1"
            "elinucb ui_X 0."
            "elinucb i_X 0."
        )
        ;;
    3)
        possibilities=(
            "epsilon_greedy u 1."
            "epsilon_greedy u 0.1"
            "epsilon_greedy i 0.1"
            "epsilon_greedy r 0.1"
            "epsilon_greedy c 0.1"
            "epsilon_greedy u 0."
            "epsilon_greedy i 0."
        )
        ;;
    4)
        possibilities=(
            "epsilon_greedy u 1."
            "epsilon_greedy u 0.1"
            "epsilon_greedy i 0.1"
            "epsilon_greedy r 0.1"
            "epsilon_greedy c 0.1"
            "elinucb u 0."
            "elinucb i 0."
            "biased_iid u 0.1"
        )
        ;;
    6)
        possibilities=(
            "epsilon_greedy uu_X 1."
            "ec ui_X 0."
            "ec rui_X 0."
            "ec comb 0."
        )
        ;;
    *)
        echo "Invalid setting: $setting"
        exit 1
        ;;
esac

# Loop over all possibilities and call the script
for entry in "${possibilities[@]}"; do
    read -r algo_name style epsilon <<< "$entry"
    ./main_driver_by_algo_style_setting.sh "$algo_name" "$style" "$epsilon" "$setting"
done

