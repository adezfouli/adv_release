Onto data:

tensorflowjs_converter --input_format keras  models/archive/learner/gonogo/onto/gonogo_learner_single_5cell/model-9300.h5 gonogo_experiment/tfjs/tfjs_learner/

tensorflowjs_converter --input_format keras nongit/archive/gonogo/onto-cv/RL_gonogo_dqn_vec/RL_nc_dqn_buf_10000_eps_0.2_lr_0.0001/model-360000.h5 gonogo_experiment/tfjs/tfjs_RL/


Round 1 adv:
For exporting the models into javascript format:

tensorflowjs_converter --input_format keras  models/archive/learner/gonogo/local-cv/round1/gonogo_learner_single_5cell/model-11800.h5 gonogo_experiment/tfjs/tfjs_learner/

tensorflowjs_converter --input_format keras nongit/archive/gonogo/local-cv/round1/RL_gonogo_dqn_vec/RL_nc_dqn_buf_100000_eps_0.05_lr_0.0001/model-1000000.h5 gonogo_experiment/tfjs/tfjs_RL/


Round 2 adv:
For exporting the models into javascript format:

tensorflowjs_converter --input_format keras  models/archive/learner/gonogo/local-cv/round2/gonogo_learner_cells_5/model-13500.h5 gonogo_experiment/tfjs/tfjs_learner/

tensorflowjs_converter --input_format keras nongit/archive/gonogo/local-cv/round2/RL_gonogo_dqn_vec/RL_nc_dqn_buf_100000_eps_0.01_lr_0.0001/model-900000.h5 gonogo_experiment/tfjs/tfjs_RL/


Round 3 adv stochastic:
For exporting the models into javascript format:

tensorflowjs_converter --input_format keras  models/archive/learner/gonogo/sreg/gonogo_learner_cells_sreg_8/model-11600.h5 gonogo_experiment/tfjs/tfjs_learner/

tensorflowjs_converter --input_format keras nongit/archive/gonogo/state-reg/RL_gonogo_dqn_vec_with_rew/RL_gonogo_2layers_0.1ent_256units_lof0.0/p-model-8500.h5 gonogo_experiment/tfjs/tfjs_RL/


Static ADV:
tensorflowjs_converter --input_format keras  models/archive/learner/gonogo/sreg/gonogo_learner_cells_sreg_8/model-11600.h5 gonogo_experiment/tfjs/tfjs_learner/

tensorflowjs_converter --input_format keras nongit/archive/gonogo/state-reg/RL_gonogo_static-1/RL_gonogo_2layers_0.01ent_256units_lof0.0/p-model-2000.h5 gonogo_experiment/tfjs/tfjs_RL/
