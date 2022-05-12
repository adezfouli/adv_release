import pandas as pd
import numpy as np
if __name__ == '__main__':
    from os import listdir
    from os.path import isfile, join

    base_path = '../nongit/archive/mrtt/RND/human_data/r5-adv-rnd/'
    path_files = base_path + 'rmrtt_data_sbj/'
    onlyfiles = [f for f in listdir(path_files) if isfile(join(path_files, f))]

    to_pays = []
    for f_path in onlyfiles:
        ll = pd.read_csv(path_files + f_path, dtype=str)
        if (not ll['stimulus'].iloc[-1].startswith('<div class = centerbox><p class = '
                                               'center-block-text>Thanks for completing this '
                                               'task!</p><p class = center-block-text>You '
                                               'earned 0 units')):
            ss = ll['stimulus'].iloc[-1]
            ss = ss.strip(('<div class = centerbox><p class = '
                                               'center-block-text>Thanks for completing this '
                                               'task!</p><p class = center-block-text>You '
                                               'earned'))

            to_pays.append({'id': ll['payment_id'].iloc[-1], 'amount': int(ss.split('unit')[0])})


    ll = pd.read_csv(base_path + 'Batch_4132088_batch_results cut.csv', dtype=str)


    # for getting who was not paid
    output_str = ''
    total_spending  = 0
    total_subjects = 0
    amount = 4.20
    not_paid = []
    for i in range(ll.shape[0]):
        paid = False
        for j in range(len(to_pays)):
            if ll['Answer.surveycode'][i] == to_pays[j]['id']:
                paid = True
        if not paid:

            total_subjects += 1
            total_spending += amount
            output_str += 'aws mturk send-bonus --worker-id ' + ll['WorkerId'].iloc[i] \
                          + ' --bonus-amount ' + "{:.2f}".format(amount) \
                          + ' --assignment-id ' + ll['AssignmentId'].iloc[i] + \
                          ' --reason ' + '\'points in the game\'; '


    # output_str = ''
    # total_spending  = 0
    # total_subjects = 0
    # for i in range(len(to_pays)):
    #     locs = ll.loc[ll['Answer.surveycode'] == to_pays[i]['id']]
    #     if locs.shape[0] == 1:
    #         total_subjects += 1
    #         total_spending += float(0.01) * (to_pays[i]['amount'])
    #         output_str += 'aws mturk send-bonus --worker-id ' + locs['WorkerId'].iloc[0] \
    #                      +' --bonus-amount ' +  "{:.2f}".format(float(0.01) * (to_pays[i]['amount'])) \
    #                     + ' --assignment-id ' + locs['AssignmentId'].iloc[0] + \
    #                     ' --reason ' + '\'points in the game\'; '
    print(output_str)
    print(total_spending)
    print(total_subjects)

    # For paying everyone the same amount of money
    # ll = pd.read_csv('../nongit/archive/mrtt/Read/human_data/r2-adv-rnd/Batch_4130454_batch_results.csv', dtype=str)
    # output_str = ''
    # total_spending  = 0
    # total_subjects = 0
    # payment = 2.75
    # for i in range(ll.shape[0]):
    #     wid = ll['WorkerId'][i]
    #     assi_id = ll['AssignmentId'][i]
    #     total_subjects += 1
    #     total_spending += payment
    #     output_str += 'aws mturk send-bonus --worker-id ' + wid \
    #                   + ' --bonus-amount ' + str(payment) \
    #                   + ' --assignment-id ' + assi_id + \
    #                   ' --reason ' + '\'points in the game\'; '
