import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf

import keras
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img

class Dataset():

    def __init__(self, image_shape, test_csv):
        self.image_shape = image_shape
        self.test_csv = test_csv

    def get_batch(self):
        path, x_test, y_test = self.load_data(self.test_csv)

        return path, x_test, y_test

    def load_data(self, path_csv):
        X = []
        Y = []
        all_image_path_list = []

        label_df = pd.read_csv(path_csv)

        imagedir_path_list = label_df['path'].values.tolist()

        read_num = 1
        for dir_nth, imagedir_n in enumerate(imagedir_path_list):
            image_path_list = os.listdir(imagedir_n)

            for image_n in image_path_list:
                sys.stdout.write("\r data_num : %d" % read_num)
                sys.stdout.flush()

                image_path = imagedir_n + image_n
                image = img_to_array(load_img(image_path, target_size=(self.image_shape[0],self.image_shape[1])))
                norm_image = image / 255.0
                X.append(norm_image)

                set_label = label_df.at[dir_nth, 'volume']
                Y.append(set_label)

                all_image_path_list.append(image_path)

                read_num += 1
                
                if read_num >= 30000:
                    break
        X = np.asarray(X)
        Y = np.asarray(Y)

        return all_image_path_list, X, Y

def save_yyplot(y,t,save_dir):
    plt.figure(figsize=(8,6))
    plt.scatter(y,t)
    plt.plot([0,1500],[0,1500],color='red', linestyle='solid')
    plt.plot([100,1500],[0,1400],color='black', linestyle='solid')
    plt.plot([0,1400],[100,1500],color='black', linestyle='solid')
    plt.plot([200,1500],[0,1300],color='grey', linestyle='solid')
    plt.plot([0,1300],[200,1500],color='grey', linestyle='solid')
    plt.plot([300,1500],[0,1200],color='black', linestyle='solid')
    plt.plot([0,1200],[300,1500],color='black', linestyle='solid')
    plt.plot([400,1500],[0,1100],color='grey', linestyle='solid')
    plt.plot([0,1100],[400,1500],color='grey', linestyle='solid')
    plt.plot([500,1500],[0,1000],color='black', linestyle='solid')
    plt.plot([0,1000],[500,1500],color='black', linestyle='solid')
    plt.plot([600,1500],[0,900],color='grey', linestyle='solid')
    plt.plot([0,900],[600,1500],color='grey', linestyle='solid')
    plt.plot([700,1500],[0,800],color='black', linestyle='solid')
    plt.plot([0,800],[700,1500],color='black', linestyle='solid')
    plt.xlim(0,1500)
    plt.ylim(0,1500)
    plt.xlabel('estimated volume')
    plt.ylabel('correct volume')
    plt.plot()
    plt.savefig(save_dir+'yyplot.png')

def save_report(path, y, t, save_dir):
    data_num = len(path)
    all_diff = [['path','predicted','true','difference']]
    diff_image_list = [['path','predicted','true','difference']]
    max_diff = 0.0
    min_diff = 1500.0
    diff_cnt10 = 0
    diff_cnt50 = 0
    diff_cnt100 = 0
    diff_cnt150 = 0
    diff_cnt200 = 0
    for n in range(len(y)):
        diff = abs(y[n][0] - t[n])
        all_diff.append([path[n],y[n][0],t[n],diff])

        if diff >= 200:
            diff_image_list.append([path[n],y[n][0],t[n],diff])

        if max_diff < diff:
            max_diff = diff
        if min_diff > diff:
            min_diff = diff

        if diff <= 10:
            diff_cnt10 += 1
        if diff <= 50:
            diff_cnt50 += 1
        if diff <= 100:
            diff_cnt100 += 1
        if diff <= 150:
            diff_cnt150 += 1
        if diff <= 200:
            diff_cnt200 += 1

    mae = mean_absolute_error(t,y)
    rmse = np.sqrt(mean_squared_error(t,y))

    acc10 = diff_cnt10 / data_num * 100
    acc50 = diff_cnt50 / data_num * 100
    acc100 = diff_cnt100 / data_num * 100
    acc150 = diff_cnt150 / data_num * 100
    acc200 = diff_cnt200 / data_num * 100

    result = [
        ['MAE','RMSE','MAE/RMSE','max difference','min difference','accuracy(10)','accuracy(50)','accuracy(100)','accuracy(150)','accuracy(200)'],
        [mae,rmse,mae/rmse,max_diff,min_diff,acc10,acc50,acc100,acc150,acc200]
    ]

    with open(save_dir+'all_diff.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(all_diff)
    with open(save_dir+'large_diff_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(diff_image_list)
    with open(save_dir+'result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(result)


save_dir = './../../Data/Report/'
model_path = './../../Data/Models/model_842_0.10.h5'
test_csv = './../../Data/Dataset_csv/test_standard.csv'
norm_param = './../../Data/Dataset_csv/standard_param.csv'

# load model
model = load_model(model_path)
image_shape = (256, 128, 3)

# load dataset
dataset = Dataset(image_shape, test_csv)
path, x_test, y_test = dataset.get_batch()

# load norm param
nparam_df = pd.read_csv(norm_param)
m = nparam_df.at[0, 'mean']
s = nparam_df.at[0, 'std']

# predict
predicted = model.predict(x_test, batch_size=1024)

# convert real scale
predicted = predicted * s + m
y_test = y_test * s + m
print(predicted)
print(y_test)

# save yyplot
save_yyplot(predicted, y_test, save_dir)

# save result_csv
save_report(path, predicted, y_test, save_dir)

