#!/bin/bash
for i in {1..7}
do
	python wandb_sweep.py -f "configs/config_06_optimized_scratch_best${i}.json"  
done 