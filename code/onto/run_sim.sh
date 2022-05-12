#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem=20gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --gres=gpu:0              # Number of GPUs (per node)

cd ~/rnn_adv/code/
source venv/bin/activate
python -m onto.sim_gonogo_dqn