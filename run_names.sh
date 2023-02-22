#!/bin/bash
configs=(best5 fast1)
for config in "${configs[@]}"
do
    python wandb_sweep.py -f "configs/config_12_optimized_transfer_${config}.json"
    for i in {1..3}
    do
        python wandb_sweep.py -f "configs/config_14_optimized_transfer_${config}_aug${i}.json"
    done
done 