import h5py
import pandas as pd
import numpy as np

from data_process import DataProcess
from util import DLogger


class DataReader:
    def __init__(self):
        pass

    @staticmethod
    def read_synth_nc():
        data = pd.read_csv("../data/synth/ql_nc.csv", header=0, sep=',', quotechar='"', keep_default_na=False)
        DLogger.logger().debug("read data from " + "../data/synth/ql_nc.csv")
        data['block'] = 1
        ids = data['id'].unique().tolist()
        dftr = pd.DataFrame({'id': ids, 'train': 'train'})
        tdftr = pd.DataFrame({'id': [], 'train': 'test'})
        data = DataProcess.train_test_between_subject(data, pd.concat((dftr, tdftr)), [1], values=['reward', 'action'])
        data = DataProcess.merge_data(data, vals=['reward', 'action'])['merged'][0]
        return data


    @staticmethod
    def merge_nc_data():
        from os import listdir
        from os.path import isfile, join
        path = "../data/nc/1_6_vs_2_6/"
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

        all_data = []
        for f in onlyfiles:
            data = pd.read_csv(path + f, header=0, sep=',', quotechar='"', keep_default_na=False)
            all_data.append(pd.DataFrame({'action': [0 if x == ' LEFT' else 1 for x in data[' side_choice']],
                                          'reward': data[' observed_reward'],
                                          'id': f,
                                          'block': 1,
                                          'schedule_type': data[' schedule_type']
                                          }))

        pd.concat(all_data).to_csv("../data/nc/merged_dynamic.csv")

        all_data = []
        for rnd in range(0, 20):
            path = "../data/nc/random_" + str(rnd) + '/'
            onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
            for f in onlyfiles:
                data = pd.read_csv(path + f, header=0, sep=',', quotechar='"', keep_default_na=False)
                all_data.append(pd.DataFrame({'action': [0 if x == ' LEFT' else 1 for x in data[' side_choice']],
                                              'reward': data[' observed_reward'],
                                              'id': f,
                                              'block': 1,
                                              'schedule_type': data[' schedule_type']
                                              }))

        pd.concat(all_data).to_csv("../data/nc/merged_static.csv")

    @staticmethod
    def read_nc_data(type=None):
        np.random.seed(1010)
        path = "../data/nc/merged_dynamic.csv"
        data_dynamic = pd.read_csv(path, header=0, sep=',', quotechar='"', keep_default_na=False)

        if type == 'dynamic':
            data = data_dynamic

        path = "../data/nc/merged_static.csv"
        data_static = pd.read_csv(path, header=0, sep=',', quotechar='"', keep_default_na=False)

        if type == 'static':
            data = data_static

        if type is None:
            data = pd.concat((data_dynamic, data_static))

        ids = data['id'].unique().tolist()
        dftr = pd.DataFrame({'id': ids, 'train': 'train'})
        tdftr = pd.DataFrame({'id': [], 'train': 'test'})
        data = DataProcess.train_test_between_subject(data, pd.concat((dftr, tdftr)), [1], values=['reward', 'action'])
        data = DataProcess.merge_data(data, vals=['reward', 'action'])['merged'][0]

        to_switch_action = np.random.binomial(1, 0.5, data['action'].shape[0])
        data['action'][to_switch_action == 0] = 1 - data['action'][to_switch_action == 0]
        return data


    @staticmethod
    def read_gonogo():
        data = pd.read_csv("../data/go-nogo/fromNCpaper/go_nogo.csv.gz", compression='gzip', header=0, sep=',', quotechar='"')

        data['id'] = data['worker_id']
        del data['worker_id']

        data['state'] = (data['correct_response'] == -1) * 1
        data['action'] = (data['key_press'] == -1) * 1
        data['reward'] = (data['key_press'] == data['correct_response']) * 1
        data['block'] = 1
        data = data.loc[data.exp_stage == "test"]

        ids = data['id'].unique().tolist()
        dftr = pd.DataFrame({'id': ids, 'train': 'train'})
        tdftr = pd.DataFrame({'id': [], 'train': 'test'})
        data = DataProcess.train_test_between_subject(data, pd.concat((dftr, tdftr)), [1],
                                                      ['reward', 'action', 'state', 'rt'])
        data = DataProcess.merge_data(data, vals=['reward', 'action', 'state', 'rt'])['merged'][0]
        data['rt'][data['rt'] != -1] = data['rt'][data['rt'] != -1] / 1000
        data['rt'] = np.concatenate((-1 * np.ones((data['rt'].shape[0]))[:, np.newaxis], data['rt'][:, :-1]), axis=1)
        data['state'] = np.concatenate(
            ((data['state'][:, :, np.newaxis] == 0) * 1, (data['state'][:, :, np.newaxis] == 1) * 1), axis=2)

        filter_sub = (1 - data['reward']).sum(axis=1) <= 22
        data['reward'] = data['reward'][filter_sub,]
        data['action'] = data['action'][filter_sub,]
        data['state'] = data['state'][filter_sub,]
        data['rt'] = data['rt'][filter_sub,]
        data['seq_lengths'] = data['seq_lengths'][filter_sub,]
        # data['state'] = np.concatenate((
        #     data['rt'][:, :, np.newaxis],
        #     data['state'][:, :, np.newaxis],
        #     np.concatenate((0 * np.ones((data['state'].shape[0], 1, 1)), data['state'][:, :, np.newaxis][:, :-1]), axis=1),
        # ), axis=2)
        return data


    @staticmethod
    def read_gonogo_random_local():
        data = pd.read_csv("../data/go-nogo/local/state-reg/merged.csv.zip", compression='zip', header=0, sep=',', quotechar='"')

        data['id'] = data['worker_id']
        del data['worker_id']

        data['state'] = (data['correct_response'] == -1) * 1
        data['action'] = (data['key_press'] == -1) * 1
        data['reward'] = (data['key_press'] == data['correct_response']) * 1
        data['block'] = 1
        data = data.loc[data.exp_stage == "test"]

        ids = data['id'].unique().tolist()
        dftr = pd.DataFrame({'id': ids, 'train': 'train'})
        tdftr = pd.DataFrame({'id': [], 'train': 'test'})
        data = DataProcess.train_test_between_subject(data, pd.concat((dftr, tdftr)), [1],
                                                      ['reward', 'action', 'state', 'rt'])
        data = DataProcess.merge_data(data, vals=['reward', 'action', 'state', 'rt'])['merged'][0]
        data['rt'][data['rt'] != -1] = data['rt'][data['rt'] != -1] / 1000
        data['rt'] = np.concatenate((-1 * np.ones((data['rt'].shape[0]))[:, np.newaxis], data['rt'][:, :-1]), axis=1)
        data['state'] = np.concatenate(
            ((data['state'][:, :, np.newaxis] == 0) * 1, (data['state'][:, :, np.newaxis] == 1) * 1), axis=2)

        filter_sub = (1 - data['reward']).sum(axis=1) <= 31.75
        data['reward'] = data['reward'][filter_sub,]
        data['action'] = data['action'][filter_sub,]
        data['state'] = data['state'][filter_sub,]
        data['rt'] = data['rt'][filter_sub,]
        data['seq_lengths'] = data['seq_lengths'][filter_sub,]
        data['id'] = [data['id'][i] for i in range(len(data['id'])) if filter_sub[i]]
        return data

    @staticmethod
    def read_MRTT_Read():
        np.random.seed(1010)
        path = "../data/MRTT/data_Read_summ.csv"
        data = pd.read_csv(path)
        data['block'] = 1
        ids = data['id'].unique().tolist()
        dftr = pd.DataFrame({'id': ids, 'train': 'train'})
        tdftr = pd.DataFrame({'id': [], 'train': 'test'})
        data = DataProcess.train_test_between_subject(data, pd.concat((dftr, tdftr)), [1], values=['reward', 'action'])
        data = DataProcess.merge_data(data, vals=['reward', 'action'])['merged'][0]

        # if you need discrete actions like me :)
        discr_actions = np.floor(np.clip(data['action'] - 0.001, a_min=0, a_max=np.inf) /  4).astype(int)
        data['action'] = discr_actions
        return data


    @staticmethod
    def read_MRTT_RND():
        np.random.seed(1010)
        path = "../data/MRTT/r25_rnd.csv"
        data = pd.read_csv(path)
        data['block'] = 1
        ids = data['id'].unique().tolist()
        dftr = pd.DataFrame({'id': ids, 'train': 'train'})
        tdftr = pd.DataFrame({'id': [], 'train': 'test'})
        data = DataProcess.train_test_between_subject(data, pd.concat((dftr, tdftr)), [1], values=['reward', 'action'])
        data = DataProcess.merge_data(data, vals=['reward', 'action'])['merged'][0]

        # if you need discrete actions like me :)
        discr_actions = np.floor(np.clip(data['action'] - 0.001, a_min=0, a_max=np.inf) /  4).astype(int)
        data['action'] = discr_actions
        return data

if __name__ == '__main__':
    # DataReader.merge_nc_data()
    # d = DataReader.read_gonogo()
    # DataReader.read_nc_data()
    DataReader.read_MRTT_Read()
    pass
