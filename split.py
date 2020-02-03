import glob
import json
from random import shuffle
from random import randint
import numpy as np
import cv2



folders = glob.glob('data/*')
folders = np.sort(folders)
print folders
label_dct = {}
for label, folder in enumerate(folders):
    pos = folder.rfind('/') +1
    label_dct[label] = folder[pos:]
json = json.dumps(label_dct)
f = open("labels.json","w")
f.write(json)
f.close()

train_list = []
val_list = []

for ii, folder in enumerate(folders):
    fnames = glob.glob(folder +'/*')
    shuffle(fnames)
    nfiles = min(len(fnames), 3000)
    fnames = fnames[:nfiles]
    for fname in fnames:
        im = cv2.imread(fname)
        try:
            a = im+2
        except:
            continue
        ss = fname + ' ' + str(ii) + '\n'
        if randint(0, 4)<1:
            val_list.append(ss)
        else:
            train_list.append(ss)

shuffle(train_list)
fp_train = open('train.lst', 'w')
for line in train_list:
    fp_train.write(line)
fp_train.close()

shuffle(val_list)
fp_val = open('val.lst', 'w')
for line in val_list:
    fp_val.write(line)
fp_val.close()
