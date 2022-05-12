#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --mem=10gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --gres=gpu:0              # Number of GPUs (per node)

cd ~/rnn_adv/code/
source venv/bin/activate
python -m nc.sim_ql_dqn_rnn