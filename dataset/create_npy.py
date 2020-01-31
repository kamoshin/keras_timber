import numpy as np
import pandas as pd
import sys
import os

from keras.preprocessing.image import array_to_img, img_to_array, load_img


class Dataset():

    def __init__(self, image_shape, train_csv, test_csv):
        self.image_shape = image_shape
        self.train_csv = train_csv
        self.test_csv = test_csv

    def get_batch(self):
        x_train, y_train = self.load_data(self.train_csv)
        print('\ntrain data num :', len(x_train))
        x_test, y_test = self.load_data(self.test_csv)
        print('\ntest data num :', len(x_test))

        return x_train, y_train, x_test, y_test

    def load_data(self, path_csv):
        X = []
        Y = []

        label_df = pd.read_csv(path_csv)

        imagedir_path_list = label_df['path'].values.tolist()

        read_num = 1
        for dir_nth, imagedir_n in enumerate(imagedir_path_list):
            image_path_list = os.listdir(imagedir_n)

            for image_n in image_path_list:
                sys.stdout.write("\r data_num : %d" % read_num)
                sys.stdout.flush()
                #imagedirの中にある画像を順に読み取り、target_size(256, 128, 3)にリサイズ
                image = img_to_array(load_img(imagedir_n+image_n, target_size=(self.image_shape[0],self.image_shape[1])))
                norm_image = image / 255.0
                X.append(norm_image)

                set_label = label_df.at[dir_nth, 'volume']
                Y.append(set_label)

                read_num += 1
                
                if read_num >= 30000:
                    break
        X = np.asarray(X)
        Y = np.asarray(Y)

        return X, Y

train_csv = './../../Data/Dataset_csv/train_standard.csv'
test_csv = './../../Data/Dataset_csv/test_standard.csv'
image_shape = (256, 128, 3)
save_dir = './../../Data/Dataset_npy/'

dataset = Dataset(image_shape,train_csv,test_csv)

x_train, y_train, x_test, y_test = dataset.get_batch()
print(x_train)
print(x_test)
print(y_train)
print(y_test)

np.save(save_dir+'x_train.npy', x_train)
np.save(save_dir+'x_test.npy', x_test)
np.save(save_dir+'y_train.npy', y_train)
np.save(save_dir+'y_test.npy', y_test)
