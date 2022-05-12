#!/bin/bash
#SBATCH --time=50:00:00
#SBATCH --mem=2gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0              # Number of GPUs (per node)

cd ~/rnn_adv/code/
source venv/bin/activate
echo "batch $1"
python -m nc.train_adv_ql_dqn $1