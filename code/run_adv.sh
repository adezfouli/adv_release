#!/bin/bash
#SBATCH --time=50:00:00
#SBATCH --mem=5gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --gres=gpu:0              # Number of GPUs (per node)

cd ~/rnn_adv/code/
source venv/bin/activate
echo "batch $1"
python -m train_adv $1