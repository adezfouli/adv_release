#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=1gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0              # Number of GPUs (per node)

cd ~/rnn_adv/code/
source venv/bin/activate
echo "batch $1"
python -m onto.train_gonogo $1