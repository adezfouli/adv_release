#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --mem=4gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0              # Number of GPUs (per node)

cd ~/rnn_adv/code/
source venv/bin/activate
echo "batch $1"
python -m mrtt.train_learner $1