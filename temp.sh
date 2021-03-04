#!/bin/bash

./py-sbatch.sh -m scripts.save_attributions
./py-sbatch.sh main.py train -n exp_b2b_reg_attr_t --checkpoints checkpoints/exp_b2b_reg_attr_t -am attributions_weighted -fm true
./py-sbatch.sh main.py train -n exp_b2b_reg_attr_p --checkpoints checkpoints/exp_b2b_reg_attr_p -am attributions_100_predicted
./py-sbatch.sh main.py train -n exp_bart_b_reg_attr_t --checkpoints checkpoints/exp_bart_b_reg_attr_t -am attributions_weighted -fm true --model-type bart --model-name facebook/bart-base
./py-sbatch.sh main.py train -n exp_bart_b_reg_attr_p --checkpoints checkpoints/exp_bart_b_reg_attr_p -am attributions_100_predicted --model-type bart --model-name facebook/bart-base
