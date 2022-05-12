from tensorflow_core.python.keras.saving import load_model

if __name__ == '__main__':
    learner_model = load_model('../nongit/archive/gonogo/state-reg/RL_gonogo_static-2/RL_gonogo_2layers_0.005014ent_256units_lof0.1/p-model-0.h5', compile=False)
    print(learner_model.summary())
