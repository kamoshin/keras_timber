import pandas as pd
import os
import shutil
import csv
import random

def makeDir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

def main():
    area_list = [['01','番号材積対応表_01田岸.xlsx']
                ,['02','番号材積対応表_02上涌波A.xlsx']
                ,['03','番号材積対応表_03上涌波S56_A.xlsx']
                ,['04','番号材積対応表_04上涌波.xlsx']
                ,['05','番号材積対応表_05上涌波S56.xlsx']
                ,['06','番号材積対応表_06上涌波S60.xlsx']
                ,['07','番号材積対応表_07田岸.xlsx']
                ,['08','番号材積対応表_08田岸ラ部02.xlsx']
                ,['09','番号材積対応表_09徳田大津.xlsx']
                ,['10','番号材積対応表_10田岸.xlsx']
                ,['11','番号材積対応表_11田岸ラ部04.xlsx']
                ,['12','番号材積対応表_12中島二部01.xlsx']
                ,['13','番号材積対応表_13中島二部02.xlsx']
                ,['14','番号材積対応表_14中島二部03.xlsx']
                ,['15','番号材積対応表_15穴水平野.xlsx']
                ,['16','番号材積対応表_16中島二部.xlsx']
                ,['17','番号材積対応表_17中島二部.xlsx']
                ,['19','番号材積対応表_19中島二部.xlsx']
                ,['20','番号材積対応表_20小松西俣01.xlsx']
                ]

    """
    area_list = [['01','番号材積対応表_01田岸.xlsx']
                ,['02','番号材積対応表_02上涌波A.xlsx']
                ,['03','番号材積対応表_03上涌波S56_A.xlsx']
                ,['04','番号材積対応表_04上涌波.xlsx']
                ,['05','番号材積対応表_05上涌波S56.xlsx']
                ,['06','番号材積対応表_06上涌波S60.xlsx']
                ,['07','番号材積対応表_07田岸.xlsx']
                ,['08','番号材積対応表_08田岸ラ部02.xlsx']
                ,['09','番号材積対応表_09徳田大津.xlsx']
                ,['10','番号材積対応表_10田岸.xlsx']
                ,['11','番号材積対応表_11田岸ラ部04.xlsx']
                ,['12','番号材積対応表_12中島二部01.xlsx']
                ,['13','番号材積対応表_13中島二部02.xlsx']
                ,['14','番号材積対応表_14中島二部03.xlsx']
                ,['15','番号材積対応表_15穴水平野.xlsx']
                ,['16','番号材積対応表_16中島二部.xlsx']
                ,['17','番号材積対応表_17中島二部.xlsx']
                ,['19','番号材積対応表_19中島二部.xlsx']
                ,['20','番号材積対応表_20小松西俣.xlsx']
                ]
    """
    
    original_Dataset_dir = 'C:/Users/kamoi/Documents/self_development/python/keras_timber/original_Dataset/'
    image_dir = original_Dataset_dir + 'image/'
    xlsx_dir = original_Dataset_dir + 'reference_volume/'
    save_dir = original_Dataset_dir + 'dataset_csv/'
    makeDir(save_dir)
    
    skip_num = [0,1,2]
    use_num = [0,1]
    sortedlist = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

    for area_n in area_list:
        area = area_n[0]
        xlsx = area_n[1]

        df = pd.read_excel(xlsx_dir+ xlsx, header=None, usecols=use_num, skiprows=skip_num, skipfooter=0, dtype={0:str})
        print(df)
        point_list = df.iloc[:,[0]].values.tolist()
        volume_list = df.iloc[:,[1]].values.tolist()
        print(point_list)
        print(volume_list)
        #exit()

        for n, volume in enumerate(volume_list):
            volume = volume[0]    #材積量
            level = int(volume // 100)    #材積量のレベル（インデックス）
            sortedlist[level].append([image_dir+area+'/'+point_list[n][0]+'/', volume])
    #print(sortedlist)
    #print(len(sortedlist))
    cnt = 0
    for n in sortedlist:
        cnt += len(n)
        #print(len(n))
    #print(cnt)
    #exit()

    for n in sortedlist:
        random.shuffle(n)

    test_num = []
    for nth,vol_n in enumerate(sortedlist):
        cnt = 0
        for data_n in vol_n:
            img_list = os.listdir(data_n[0])
            cnt += len(img_list)
        test_num.append(cnt*0.3)
        #print(nth,cnt,cnt*0.3)
    #print(test_num)

    test_volume_list = [['path','volume']]
    train_volume_list = [['path','volume']]
    for nth,vol_n in enumerate(sortedlist):
        cnt = 0
        for data_n in vol_n:
            img_list = os.listdir(data_n[0])
            cnt += len(img_list)
            if test_num[nth] > cnt:
                test_volume_list.append(data_n)
            else:
                train_volume_list.append(data_n)
    #print(test_volume_list)
    #print(len(test_volume_list))
    #print(train_volume_list)
    #print(len(train_volume_list))

    with open(save_dir+'test_no_standard.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(test_volume_list)
    with open(save_dir+'train_no_standard.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(train_volume_list)

if __name__ == '__main__':
    main()