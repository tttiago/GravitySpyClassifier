#!/bin/bash
for i in {2..3}
do
    python wandb_sweep.py -f "configs/config_14_optimized_transfer_fast1_aug${i}.json"  
done 
python wandb_sweep.py -i i3ewiaec