"""
@Created Time : 2022/12/10
@Author  : LiYao
@FileName: CNNBehaviorExtractor.py
@Description:nfstream提取行为特征
@Modified:
    :First modified
    :Modified content:
"""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import optim, nn
import torch.nn.functional as F
# from torchvision import transforms
from torch.utils.data.dataset import Dataset
import random


class DatasetFromCSV(Dataset):
    def __init__(self,):
        self.data = pd.read_csv('./merge_data_mit_5labels.csv')


    def cal(self):
        src_bytes_data1 = self.data.loc[self.data['label'] == 1]['udps.bi_pkt_size'].tolist()
        src_bytes_data2 = self.data.loc[self.data['label'] == 2]['udps.bi_pkt_size'].tolist()
        src_bytes_data3 = self.data.loc[self.data['label'] == 3]['udps.bi_pkt_size'].tolist()
        src_bytes_data4 = self.data.loc[self.data['label'] == 4]['udps.bi_pkt_size'].tolist()
        src_bytes_data5 = self.data.loc[self.data['label'] == 5]['udps.bi_pkt_size'].tolist()

        num1 = [0 for _ in range(20)]
        num2 = [0 for _ in range(20)]
        num3 = [0 for _ in range(20)]
        num4 = [0 for _ in range(20)]
        num5 = [0 for _ in range(20)]

        for index,val in enumerate(src_bytes_data1):
            if val != val:
                continue
            str_value1 = val.replace(' ', '')
            for pad_i in range(0, 80):
                str_value1 += '0'
            for value_index_sl2 in range(0, 20 * 2, 2):
                temp_value1 = str_value1[value_index_sl2] + str_value1[value_index_sl2 + 1]
                temp_value1 = int(temp_value1)
                num1[int(value_index_sl2 / 2)] += temp_value1

        for index,val in enumerate(src_bytes_data2):
            if val != val:
                continue
            str_value2 = val.replace(' ', '')
            for pad_i in range(0, 80):
                str_value2 += '00'
            for value_index_sl2 in range(0, 20 * 2, 2):
                temp_value1 = str_value2[value_index_sl2] + str_value2[value_index_sl2 + 1]
                temp_value1 = int(temp_value1)
                num2[int(value_index_sl2 / 2)] += temp_value1


        for index,val in enumerate(src_bytes_data3):
            if val != val:
                continue
            str_value3 = val.replace(' ', '')
            for pad_i in range(0, 80):
                str_value3 += '00'
            for value_index_sl2 in range(0, 20 * 2, 2):
                temp_value1 = str_value3[value_index_sl2] + str_value3[value_index_sl2 + 1]
                temp_value1 = int(temp_value1)
                num3[int(value_index_sl2 / 2)] += temp_value1

        for index,val in enumerate(src_bytes_data4):
            if val != val:
                continue
            str_value4 = val.replace(' ', '')
            for pad_i in range(0, 80):
                str_value4 += '00'
            for value_index_sl2 in range(0, 20 * 2, 2):
                temp_value1 = str_value4[value_index_sl2] + str_value4[value_index_sl2 + 1]
                temp_value1 = int(temp_value1)
                num4[int(value_index_sl2 / 2)] += temp_value1

        for index,val in enumerate(src_bytes_data5):
            if val != val:
                continue
            str_value5 = val.replace(' ', '')
            for pad_i in range(0, 80):
                str_value5 += '00'
            for value_index_sl2 in range(0, 20 * 2, 2):
                temp_value1 = str_value5[value_index_sl2] + str_value5[value_index_sl2 + 1]
                temp_value1 = int(temp_value1)
                num5[int(value_index_sl2 / 2)] += temp_value1

        scaler =MinMaxScaler(feature_range=(0,1))


        for iii in range(20):
            num1[iii]  = (num1[iii]/len(src_bytes_data1))
            num2[iii] = (num2[iii] / len(src_bytes_data2))
            num3[iii]  = (num3[iii]/len(src_bytes_data3))
            num4[iii] = (num4[iii] / len(src_bytes_data4))
            num5[iii] = (num5[iii] / len(src_bytes_data5))

        return scaler.fit_transform(np.array(num1).reshape(-1,1)),\
                scaler.fit_transform(np.array(num2).reshape(-1, 1)),\
                scaler.fit_transform(np.array(num3).reshape(-1, 1)),\
                scaler.fit_transform(np.array(num4).reshape(-1, 1)),\
                scaler.fit_transform(np.array(num5).reshape(-1, 1))




if __name__ == '__main__':
    db_obj = DatasetFromCSV()
    a,b,c,d,e = db_obj.cal()
    ll = pd.DataFrame({'a':a.reshape(1,-1).tolist()[0],'b':b.reshape(1,-1).tolist()[0],'c':c.reshape(1,-1).tolist()[0],'d':d.reshape(1,-1).tolist()[0],'e':e.reshape(1,-1).tolist()[0]})
    ll.to_csv('./pkt_len_.csv',index=False)
    print(a.reshape(1,-1).tolist()[0])
    print(b)
    print(c)
    print(d)
    print(e)




    # data = pd.read_csv('./src_data/cnn_merge_data_mit.csv')
    # src_bytes_data = data['udps.bi_transport_payload'].tolist()
    # for i in range(0, len(src_bytes_data)):
    #     str_value = src_bytes_data[i].replace(' ', '')
    #     feature_matrix = np.zeros((28, 28))
    #     num_feature = []
    #     for pad_i in range(0, 784):
    #         str_value += '00'
    #     for value_index_sl2 in range(0, 784 * 2, 2):
    #         temp_value = '0x' + str_value[value_index_sl2] + str_value[value_index_sl2 + 1]
    #         temp_value = int(temp_value, 16)
    #         num_feature.append(temp_value)

        # cnt = 0
        # for row in range(0, 28):
        #     for column in range(0, 28):
        #         feature_matrix[row][column] = num_feature[cnt]
        #         cnt += 1
        # print(feature_matrix)
        #
        # print('-----------------------')

# batch_size = 64
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize((0.5,), (0.5,))])
#
# train_data = DatasetFromCSV('./data/train.csv', 28, 28, transform)
# test_data = DatasetFromCSV("./data/test.csv", 28, 28, transform)
#
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
#
# img, lab = next(iter(train_loader))
# print(img.shape)
