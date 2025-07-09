possibilities=(
            "epsilon_greedy uu_X 1."
            "ec ui_X 0."
            "ec rui_X 0."
            "ec comb 0."
        )

# Loop over all possibilities and call the script
for entry in "${possibilities[@]}"; do
    read -r algo_name style epsilon <<< "$entry"
    ./varying_dim_runscript_by_algo_style_eps.sh "$algo_name" "$style" "$epsilon"
done

