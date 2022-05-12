import re
from enum import Enum
from random import shuffle
import pandas as pd
import numpy as np

from util import DLogger


class State(Enum):
    STAGE_NULL = -1
    SUBMIT_INV = 0
    STAGE_RESULT = 2
    STAGE_RESULT_SELF = 3
    STAGE_RESULT_OTHER = 4
    ROUND_RESULT = 6
    SCR_REPAY = 5
    INVST_RESULTS = 7
    REPAY_RESULTS = 8
    ABORT = 10
    END_EXPR=11
    START_EXPR = 12


class DataReader:

    @staticmethod
    def process_trust_data(line_content):
        event = None
        l = len(line_content)
        if l > 0:
            t = line_content[0]
            if l > 5:
                if ''.join(line_content[1:4]) == 'KEYPRESS:SUBMITINVESTMENT':
                    event = {
                        'time': t,
                        'state': [State.SUBMIT_INV, 0, 0],
                        'action': line_content[5],
                        'reward': 0}

                if ''.join(line_content[1:7]) == 'ADDGRAPHIC:{[ID+text:SHOWINGMESSAGE<Repay>ANDPIC':
                    event = {
                        'time': t,
                        'state': [State.SCR_REPAY, 0, 0],
                        'action': -1,
                        'reward': 0}
                if ''.join(line_content[1:4]) == 'SETTINGSTAGERESULTS:':
                    event = {
                        'time': t,
                        'state': [State.STAGE_RESULT, line_content[6], line_content[9]],
                        'action': -1,
                        'reward': 0}

                if ''.join(line_content[1:4]) == 'SETTINGROUNDRESULTS:':
                    event = {
                        'time': t,
                        'state': [State.ROUND_RESULT, line_content[6], line_content[10]],
                        'action': -1,
                        'reward': 0}

            if l >= 3:
                if ''.join(line_content[1:3]) == 'ABORTREQUESTED':
                    event = {
                        'time': t,
                        'state': [State.ABORT, None, None],
                        'action': None,
                        'reward': None}

                if ''.join(line_content[1:3]).startswith('ADDGRAPHIC:{[exp.complete:'):
                    event = {
                        'time': t,
                        'state': [State.END_EXPR, None, None],
                        'action': None,
                        'reward': None}

                if ''.join(line_content[1:4]) == 'ADDGRAPHIC:{[expstart:]}':
                    event = {
                        'time': t,
                        'state': [State.START_EXPR, None, None],
                        'action': None,
                        'reward': None}

        return event


    @classmethod
    def read_beh(cls, fname, process_line):

        output = []

        # start_phrase = '[--- start of experiment Trust, VERSION: 1 ---]'
        # end_phrase = '[--- end of experiment Trust ---]'

        with open(fname) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line

        for l in range(len(content)):
            event = process_line(re.split(" |\t", content[l].strip()))
            if event is not None:
                output.append(event)

        return output

    @classmethod
    def process_invesment(cls, data):
        cur_round = 0
        new_round = False
        expr_started = False
        invst_expr = False
        output_data = []
        for d in data:

            if d['state'][0] == State.START_EXPR:
                cur_round = 0
                expr_started = True

            elif d['state'][0] == State.ABORT:
                output_data = []
                expr_started = False
                invst_expr = False

            elif expr_started:
                if d['state'][0] == State.END_EXPR:
                    if expr_started:
                        break

                if d['state'][0] == State.SUBMIT_INV:
                    cur_round += 1
                    new_round = True
                    invst_expr = True

                if d['state'][0] == State.STAGE_RESULT:
                    if new_round:
                        d['state'][0] = State.INVST_RESULTS
                        new_round = False
                    else:
                        d['state'][0] = State.REPAY_RESULTS

                d['state'][1] = int(d['state'][1]) if d['state'][1] else None
                d['state'][2] = int(d['state'][2]) if d['state'][2] else None

                d['state0'] = d['state'][0]
                d['state1'] = d['state'][1]
                d['state2'] = d['state'][2]

                d['action'] = int(d['action']) if d['action'] else None
                d['round'] = cur_round

                if invst_expr:
                    output_data.append(d)

        return output_data

    @classmethod
    def read_all_files(cls, paths):
        from os import walk

        all_data = {}
        for path in paths:
            fs = []
            for (dirpath, dirnames, filenames) in walk(path):
                for file in filenames:
                    if file.endswith(".txt"):
                        fs.append(file)
                break

            DLogger.logger().debug("files for processing: %d" % len(fs))
            for f in fs:
                data = DataReader.read_beh(path + f, DataReader.process_trust_data)
                data = DataReader.process_invesment(data)

                if len(data) == 0:
                    DLogger.logger().debug('empty file {}'.format(f))
                elif len(data) != 50:
                    DLogger.logger().debug('unrecognized length %d of events in file %s' % (len(data), f))
                elif f in data:
                    raise Exception('file name already exists: {}'.format(f))
                else:
                    subj_data = pd.DataFrame(data)
                    subj_data['id'] = f
                    all_data[f] = subj_data

        DLogger.logger().debug("files imported: %d" % len(all_data))
        return all_data

    @staticmethod
    def summarise_data(df):
        rdf = df[df.state0 == 'State.ROUND_RESULT']

        # calculating investment amount and reward from round results
        df_new = pd.DataFrame()
        df_new['action'] = (rdf.state1 + rdf.state2 - 20) / 2
        df_new['reward'] = (3 * df_new['action'] - rdf.state2)
        df_new['id'] = rdf['id']
        df_new.to_csv('../nongit/data/MRTT_Read/' + 'data_Read_summ.csv', index=False)



if __name__ == '__main__':
    # d_se = DataReader.read_all_files(
    #     ['../nongit/data/MRTT_Read/rbs_v1/',
    #      '../nongit/data/MRTT_Read/rbs_v2/',
    #      '../nongit/data/MRTT_Read/rbs_se/',
    #      '../nongit/data/MRTT_Read/uk_se/',
    #      ]
    # )
    # df = pd.concat(d_se.values())
    # df.to_csv('../nongit/data/MRTT_Read/' + 'data_Read.csv')

    df = pd.DataFrame.from_csv('../nongit/data/MRTT_Read/' + 'data_Read.csv')
    DataReader.summarise_data(df)
