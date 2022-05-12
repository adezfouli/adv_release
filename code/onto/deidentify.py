
from os import listdir
from os.path import isfile, join
mypath = "../nongit/archive/data for upload/gonogo/training/"
# mypath = "../nongit/archive/data for upload/gonogo/test/gonogo_data_2/"
onlyfiles = [mypath + f for f in listdir(mypath) if isfile(join(mypath, f))]

import csv

for fp in onlyfiles:
    with open(fp, 'rb') as f:
        lines=f.readlines()
        lines.pop()
    # if len(lines) != 365:
    #     print(fp)
    with open(fp, 'wb') as f:
        f.writelines(lines)
