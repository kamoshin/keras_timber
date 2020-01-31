import pandas as pd
import csv
import numpy as np

csv_dir = './../../original_Dataset/Dataset_csv/'

train_df = pd.read_csv(csv_dir+'train_no_standard.csv')
train_path_list = train_df['path'].values.tolist()

test_df = pd.read_csv(csv_dir+'test_no_standard.csv')
test_path_list = test_df['path'].values.tolist()

vol_list = []
for n, path in enumerate(train_path_list):
    vol_list.append(train_df.at[n, 'volume'])
"""
for n, path in enumerate(test_path_list):
    vol_list.append(test_df.at[n, 'volume'])
"""

m = np.average(vol_list)
s = np.std(vol_list)

print('average:', m)
print('stdev:', s)

train_std = [['path','volume']]
for n, path in enumerate(train_path_list):
    set_vol = (train_df.at[n, 'volume'] - m) / s
    train_std.append([path, set_vol])

with open(csv_dir+'train_standard.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(train_std)
print(train_std)

test_std = [['path','volume']]
for n, path in enumerate(test_path_list):
    set_vol = (test_df.at[n, 'volume'] - m) / s
    test_std.append([path, set_vol])

with open(csv_dir+'test_standard.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(test_std)
print(test_std)

std_param = [['mean', 'std'],[m, s]]
with open(csv_dir+'standard_param.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(std_param)
