import numpy as np
import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

"""
Updating the dataset summary csv files with the data split information.
"""

path2csv ='dataset_summary_korea_random_FL.csv'
df    = pd.read_csv(path2csv)

participant_id = df['participant_id']
mw = df['MW']
nonmw = df['non MW']
allvideos= df['num of videos']
whichset = df['set']
gender = df['gender']
glasses = df['glasses']
given_id = df['ID']
participants = list(set(participant_id))
test_fold = {}

# data split
random_state_par = 6
train_val_set, test_set = train_test_split(participant_id, train_size=0.90, random_state=random_state_par)
train_val_set = getattr(train_val_set, 'values')
train_set, val_set = train_test_split(train_val_set, train_size=0.88, random_state=random_state_par)
test_set = getattr(test_set, 'values')

# 5-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=random_state_par)
for i, (train_index, test_index) in enumerate(kf.split(train_val_set)):
    for idx in test_index:
        test_fold[train_val_set[idx]] = i

# update the csv file
outfile = open('dataset_summary_korea_random_FL_updated.csv', 'w', newline='')
writer = csv.writer(outfile)
writer.writerow(['participant_id', 'MW', 'non MW', 'num of videos', 'set', 'gender', 'glasses', 'ID', 'test_fold_num'])
for i, part in enumerate(participant_id):
    if part in train_set:
        setval = 'train'
    elif part in val_set:
        setval = 'val'
    else:
        setval = 'test'
    if part not in test_fold:
        tfold = 5
    elif allvideos[i] < 4:
        tfold = 'None'
    else:
        tfold = test_fold[part]
    data = [part, mw[i], nonmw[i], allvideos[i], setval, gender[i], glasses[i], given_id[i], tfold]
    writer.writerow(data)
outfile.close()


