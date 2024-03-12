"""
@Created Time : 2022/11/29
@Author  : LiYao
@FileName: NewGraphDataset.py
@Description:图神经网络输入，新构图法
@Modified:
    :First modified
    :Modified content:040312整理上传
"""
import copy
import random

import math
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import pandas as pd
import numpy as np

class GraphMitDataset_bak(InMemoryDataset):
    def __init__(self, root, label1, label2, label3, label4, label5, node_counts, transform=None,
                 pre_transform=None):
        self.label1 = label1
        self.label2 = label2
        self.label3 = label3
        self.label4 = label4
        self.label5 = label5
        self.node_counts = node_counts
        self.var_size = 5
        self.src_data = pd.read_csv('../PublicSrcdata/merge_data_mit_5labels.csv')
        self.break_num = 100000
        # 数据的下载和处理过程在父类中调用实现
        super(GraphMitDataset_bak, self).__init__(root, transform, pre_transform)
        # 加载数据
        self.data, self.slices = torch.load(self.processed_paths[0])

    # 将函数修饰为类属性
    @property
    def raw_file_names(self):
        return ['file_1', 'file_2']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def node_features(self, floor_num: int, direction: [], pkt_size: [], same_floor_dict: {}):
        """添加节点特征
        同层个数，上层突发个数，下层突发个数，上/同层包个数比，下/同层包个数比
        包长度，同层包总长度，上层包总长度，下层包总长度
        包长度同层占比，字节/同层包个数（平均包字节数）
        """
        # 每张图的特征向量
        pic_features_vec = []
        # 每个节点的特征向量
        node_vec = []
        # 同层包个数，上层包个数，下层包个数，同层字节数，上层字节数，下层字节数
        same_floor_num = 0
        last_floor_num = 0
        next_floor_num = 0
        same_floor_size = 0
        last_floor_size = 0
        next_floor_size = 0

        for floor in range(0, floor_num + 1):
            if floor == 0:
                # 获取该层每个节点的index
                for node_same in same_floor_dict[floor]:
                    same_floor_size += int(pkt_size[node_same])  # 计算该层字节总数
                if floor_num != 0:
                    for node_next in same_floor_dict[floor + 1]:
                        next_floor_size += int(pkt_size[node_next])  # 计算下层字节总数
                    next_floor_num = len(same_floor_dict[floor + 1])  # 下层节点数
                same_floor_num = len(same_floor_dict[floor])  # 计算该层总节点数

                '''添加节点特征
                同层个数，上层突发个数，下层突发个数，上/同层包个数比，下/同层包个数比
                包长度，同层包总长度，上层包总长度，下层包总长度
                包长度同层占比，字节/同层包个数（平均包字节数）
                '''
                for node in same_floor_dict[floor]:
                    if direction[node] == '1':
                        node_vec.append(int(pkt_size[node]))
                    elif direction[node] == '0':
                        node_vec.append(0 - int(pkt_size[node]))
                    node_vec.append(same_floor_num)
                    node_vec.append(last_floor_num)
                    node_vec.append(next_floor_num)
                    node_vec.append(last_floor_num / same_floor_num)
                    node_vec.append(next_floor_num / same_floor_num)
                    node_vec.append(same_floor_size)
                    node_vec.append(last_floor_size)
                    node_vec.append(next_floor_size)
                    node_vec.append(int(pkt_size[node]) / same_floor_size)
                    node_vec.append(same_floor_size / same_floor_num)

                    # 0329 add feature
                    if last_floor_size  == 0:
                        node_vec.append(0)
                    else:
                        node_vec.append(same_floor_size / last_floor_size)
                    if next_floor_size == 0:
                        node_vec.append(0)
                    else:
                        node_vec.append(same_floor_size / next_floor_size)

                    # 添加到特征矩阵中
                    pic_features_vec.append(copy.deepcopy(node_vec))
                    node_vec.clear()
            elif floor == floor_num:
                # 获取该层每个节点的index
                for node_same in same_floor_dict[floor]:
                    same_floor_size += int(pkt_size[node_same])  # 计算该层字节总数
                for node_last in same_floor_dict[floor - 1]:
                    last_floor_size += int(pkt_size[node_last])  # 计算上层字节总数

                same_floor_num = len(same_floor_dict[floor])  # 计算该层总节点数
                last_floor_num = len(same_floor_dict[floor - 1])  # 上层节点数
                '''添加节点特征
                同层个数，上层突发个数，下层突发个数，上/同层包个数比，下/同层包个数比
                包长度，同层包总长度，上层包总长度，下层包总长度
                包长度同层占比，字节/同层包个数（平均包字节数）
                '''
                for node in same_floor_dict[floor]:
                    if direction[node] == '1':
                        node_vec.append(int(pkt_size[node]))
                    elif direction[node] == '0':
                        node_vec.append(0 - int(pkt_size[node]))
                    node_vec.append(same_floor_num)
                    node_vec.append(last_floor_num)
                    node_vec.append(next_floor_num)
                    node_vec.append(last_floor_num / same_floor_num)
                    node_vec.append(next_floor_num / same_floor_num)
                    node_vec.append(same_floor_size)
                    node_vec.append(last_floor_size)
                    node_vec.append(next_floor_size)
                    node_vec.append(int(pkt_size[node]) / same_floor_size)
                    node_vec.append(same_floor_size / same_floor_num)

                    # 0329 add feature
                    if last_floor_size  == 0:
                        node_vec.append(0)
                    else:
                        node_vec.append(same_floor_size / last_floor_size)
                    if next_floor_size == 0:
                        node_vec.append(0)
                    else:
                        node_vec.append(same_floor_size / next_floor_size)

                    # 添加到特征矩阵中
                    pic_features_vec.append(copy.deepcopy(node_vec))
                    node_vec.clear()
            else:
                # 获取该层每个节点的index
                for node_same in same_floor_dict[floor]:
                    same_floor_size += int(pkt_size[node_same])  # 计算该层字节总数
                for node_last in same_floor_dict[floor - 1]:
                    last_floor_size += int(pkt_size[node_last])  # 计算上层字节总数
                for node_next in same_floor_dict[floor + 1]:
                    next_floor_size += int(pkt_size[node_next])  # 计算下层字节总数

                same_floor_num = len(same_floor_dict[floor])  # 计算该层总节点数
                last_floor_num = len(same_floor_dict[floor - 1])  # 上层节点数
                next_floor_num = len(same_floor_dict[floor + 1])  # 下层节点数
                '''添加节点特征
                包长度，
                同层个数，上层突发个数，下层突发个数，上/同层包个数比，下/同层包个数比
                同层包总长度，上层包总长度，下层包总长度
                包长度同层占比，字节/同层包个数（平均包字节数）
                '''
                for node in same_floor_dict[floor]:
                    if direction[node] == '1':
                        node_vec.append(int(pkt_size[node]))
                    elif direction[node] == '0':
                        node_vec.append(0 - int(pkt_size[node]))
                    node_vec.append(same_floor_num)
                    node_vec.append(last_floor_num)
                    node_vec.append(next_floor_num)
                    node_vec.append(last_floor_num / same_floor_num)
                    node_vec.append(next_floor_num / same_floor_num)
                    node_vec.append(same_floor_size)
                    node_vec.append(last_floor_size)
                    node_vec.append(next_floor_size)
                    node_vec.append(int(pkt_size[node]) / same_floor_size)
                    node_vec.append(same_floor_size / same_floor_num)

                    # 0329 add feature
                    if last_floor_size  == 0:
                        node_vec.append(0)
                    else:
                        node_vec.append(same_floor_size / last_floor_size)
                    if next_floor_size == 0:
                        node_vec.append(0)
                    else:
                        node_vec.append(same_floor_size / next_floor_size)

                    # 添加到特征矩阵中
                    pic_features_vec.append(copy.deepcopy(node_vec))
                    node_vec.clear()
            # 一层循环完后全部归零
            same_floor_num = 0
            last_floor_num = 0
            next_floor_num = 0
            same_floor_size = 0
            last_floor_size = 0
            next_floor_size = 0

        return pic_features_vec

    def generate_new_graph(self, direction_list: [], pkt_size_list: [], label):
        """生成无向图"""
        dir_length = len(direction_list)
        # 生成邻接矩阵存边
        neighbor_matrix = np.zeros((dir_length, dir_length))
        # 突发转换标志位
        change_flag = direction_list[0]
        # 同层字典
        same_floor_dict = {}
        # 层数
        floor_num = 0
        #0329 边属性
        edge_attr_list = []
        for index in range(0, dir_length):
            # index是节点序号，这个循环等于遍历所有节点
            if index == 0:
                same_floor_dict.update({floor_num: [index]})
                continue
            if direction_list[index] == change_flag:
                same_floor_dict[floor_num].append(index)
            else:
                # 不等于前面一个的标志位说明转换了，层数+1，且标志位需要转换
                floor_num += 1
                change_flag = direction_list[index]
                same_floor_dict.update({floor_num: [index]})

        # 遍历所有层，添加边
        for i in range(0, floor_num + 1):
            # 遍历同层节点
            for seq_index, seq in enumerate(same_floor_dict[i]):
                if seq_index == 0:
                    continue
                neighbor_matrix[seq][seq - 1] = 1
                neighbor_matrix[seq - 1][seq] = 1
                edge_attr_list.append([0])
                edge_attr_list.append([0])

            if i == 0:
                continue
            for last_seq in same_floor_dict[i - 1]:
                for this_seq in same_floor_dict[i]:
                    neighbor_matrix[this_seq][last_seq] = 1
                    neighbor_matrix[last_seq][this_seq] = 1
                    edge_attr_list.append([1])
                    edge_attr_list.append([1])

        edge_index = np.argwhere(neighbor_matrix == 1).tolist()
        Y = [int(label)]
        return edge_index, Y, self.node_features(floor_num, direction_list, pkt_size_list, same_floor_dict),edge_attr_list

    def process(self):
        data_list = []  # 存放所有图的列表
        # --------以下为要建立图实例的地方--------
        data_src = self.src_data
        label1_data = data_src.loc[data_src['label'] == 1]
        label2_data = data_src.loc[data_src['label'] == 2]
        label3_data = data_src.loc[data_src['label'] == 3]
        label4_data = data_src.loc[data_src['label'] == 4]
        label5_data = data_src.loc[data_src['label'] == 5]

        # 数据包方向
        label1_pkt_direction = label1_data['udps.bi_flow_pkt_direction'].tolist()
        label2_pkt_direction = label2_data['udps.bi_flow_pkt_direction'].tolist()
        label3_pkt_direction = label3_data['udps.bi_flow_pkt_direction'].tolist()
        label4_pkt_direction = label4_data['udps.bi_flow_pkt_direction'].tolist()
        label5_pkt_direction = label5_data['udps.bi_flow_pkt_direction'].tolist()

        # 数据包大小
        label1_pkt_size = label1_data['udps.bi_pkt_size'].tolist()
        label2_pkt_size = label2_data['udps.bi_pkt_size'].tolist()
        label3_pkt_size = label3_data['udps.bi_pkt_size'].tolist()
        label4_pkt_size = label4_data['udps.bi_pkt_size'].tolist()
        label5_pkt_size = label5_data['udps.bi_pkt_size'].tolist()

        print('label1 length:' + str(len(label1_pkt_direction)))
        print('label2 length:' + str(len(label2_pkt_direction)))
        print('label3 length:' + str(len(label3_pkt_direction)))
        print('label4 length:' + str(len(label4_pkt_direction)))
        print('label5 length:' + str(len(label5_pkt_direction)))

        del label1_data
        del label2_data
        del label3_data
        del label4_data
        del label5_data

        index_cnt = 0
        for index in range(0, len(label1_pkt_direction)):
            if label1_pkt_direction[index] != label1_pkt_direction[index]:
                continue  # 如果出现空值，则直接下一条数据
            temp_list = label1_pkt_direction[index].split(' ')
            pkt_size_temp = label1_pkt_size[index].split(' ')
            temp_list.pop()
            pkt_size_temp.pop()
            if len(temp_list) < self.var_size or (np.array(temp_list[0:self.node_counts]) == '1').all() \
                    or (np.array(temp_list[0:self.node_counts]) == '0').all():
                continue
            temp_list = temp_list[0:self.node_counts]
            pkt_size_temp = pkt_size_temp[0:self.node_counts]

            edge_index, Y, x,edge_attr = self.generate_new_graph(temp_list, pkt_size_list=pkt_size_temp, label=self.label1)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            Y = torch.tensor(Y, dtype=torch.long)
            x = torch.tensor(x, dtype=torch.float)
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)

            data = Data(x=x, edge_attr=edge_attr,edge_index=edge_index.t().contiguous(), y=Y)
            data_list.append(data)
            index_cnt += 1
            if self.break_num == index_cnt:
                break
        del label1_pkt_direction
        l1 = len(data_list)
        print('label1有' + str(l1) + '条数据')

        index_cnt2 = 0
        for index2 in range(0, len(label2_pkt_direction)):
            if label2_pkt_direction[index2] != label2_pkt_direction[index2]:
                continue  # 如果出现空值，则直接下一条数据
            temp_list = label2_pkt_direction[index2].split(' ')
            pkt_size_temp = label2_pkt_size[index2].split(' ')
            temp_list.pop()
            pkt_size_temp.pop()
            if len(temp_list) < self.var_size or (np.array(temp_list[0:self.node_counts]) == '1').all() \
                    or (np.array(temp_list[0:self.node_counts]) == '0').all():
                continue
            temp_list = temp_list[0:self.node_counts]
            pkt_size_temp = pkt_size_temp[0:self.node_counts]

            edge_index, Y, x,edge_attr = self.generate_new_graph(temp_list, pkt_size_list=pkt_size_temp, label=self.label2)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            Y = torch.tensor(Y, dtype=torch.long)
            x = torch.tensor(x, dtype=torch.float)
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)

            data = Data(x=x, edge_attr=edge_attr,edge_index=edge_index.t().contiguous(), y=Y)
            data_list.append(data)
            index_cnt2 += 1
            # if self.break_num == index_cnt2:
            #     break
        del label2_pkt_direction
        l2 = len(data_list)
        print('label2有' + str(l2 - l1) + '条数据')

        index_cnt3 = 0
        for index3 in range(0, len(label3_pkt_direction)):
            if label3_pkt_direction[index3] != label3_pkt_direction[index3]:
                continue  # 如果出现空值，则直接下一条数据
            temp_list = label3_pkt_direction[index3].split(' ')
            pkt_size_temp = label3_pkt_size[index3].split(' ')
            temp_list.pop()
            pkt_size_temp.pop()
            if len(temp_list) < self.var_size or (np.array(temp_list[0:self.node_counts]) == '1').all() \
                    or (np.array(temp_list[0:self.node_counts]) == '0').all():
                continue
            temp_list = temp_list[0:self.node_counts]
            pkt_size_temp = pkt_size_temp[0:self.node_counts]

            edge_index, Y, x,edge_attr = self.generate_new_graph(temp_list, pkt_size_list=pkt_size_temp, label=self.label3)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            Y = torch.tensor(Y, dtype=torch.long)
            x = torch.tensor(x, dtype=torch.float)
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)

            data = Data(x=x, edge_attr=edge_attr,edge_index=edge_index.t().contiguous(), y=Y)
            data_list.append(data)
            index_cnt3 += 1
            if self.break_num == index_cnt3:
                break
        del label3_pkt_direction
        l3 = len(data_list)
        print('label3有' + str(l3 - l2) + '条数据')

        index_cnt4 = 0
        for index4 in range(0, len(label4_pkt_direction)):
            if label4_pkt_direction[index4] != label4_pkt_direction[index4]:
                continue  # 如果出现空值，则直接下一条数据
            temp_list = label4_pkt_direction[index4].split(' ')
            pkt_size_temp = label4_pkt_size[index4].split(' ')
            temp_list.pop()
            pkt_size_temp.pop()
            if len(temp_list) < self.var_size or (np.array(temp_list[0:self.node_counts]) == '1').all() \
                    or (np.array(temp_list[0:self.node_counts]) == '0').all():
                continue
            temp_list = temp_list[0:self.node_counts]
            pkt_size_temp = pkt_size_temp[0:self.node_counts]

            edge_index, Y, x,edge_attr = self.generate_new_graph(temp_list, pkt_size_list=pkt_size_temp, label=self.label4)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            Y = torch.tensor(Y, dtype=torch.long)
            x = torch.tensor(x, dtype=torch.float)
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)

            data = Data(x=x, edge_attr=edge_attr,edge_index=edge_index.t().contiguous(), y=Y)
            data_list.append(data)
            index_cnt4 += 1
            if self.break_num == index_cnt4:
                break
        del label4_pkt_direction
        l4 = len(data_list)
        print('label4有' + str(l4 - l3) + '条数据')

        index_cnt5 = 0
        for index5 in range(0,len(label5_pkt_direction)):
            if label5_pkt_direction[index5] != label5_pkt_direction[index5]:
                continue  # 如果出现空值，则直接下一条数据
            temp_list = label5_pkt_direction[index5].split(' ')
            pkt_size_temp = label5_pkt_size[index5].split(' ')
            temp_list.pop()
            pkt_size_temp.pop()
            # if len(temp_list) < self.var_size or (np.array(temp_list[0:self.node_counts]) == '1').all() \
            #         or (np.array(temp_list[0:self.node_counts]) == '0').all():
            if len(temp_list) < self.var_size:
                continue
            temp_list = temp_list[0:self.node_counts]
            pkt_size_temp = pkt_size_temp[0:self.node_counts]
            # self.node_counts = len(temp_list)

            edge_index, Y, x,edge_attr = self.generate_new_graph(temp_list, pkt_size_list=pkt_size_temp, label=self.label5)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            Y = torch.tensor(Y, dtype=torch.long)
            x = torch.tensor(x, dtype=torch.float)
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)

            data = Data(x=x, edge_attr=edge_attr,edge_index=edge_index.t().contiguous(), y=Y)
            data_list.append(data)
            index_cnt5 += 1
            if self.break_num == index_cnt5:
                break
        del label5_pkt_direction
        l5 = len(data_list)
        print('label5有' + str(l5 - l4) + '条数据')

        random.shuffle(data_list)
        print('共计' + str(len(data_list)) + '条数据')
        # --------以上为要建立图实例的地方--------
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_filter is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        # 这里的save方式以及路径需要对应构造函数中的load操作
        torch.save((data, slices), self.processed_paths[0])
class GraphMitDataset(InMemoryDataset):
    def __init__(self, root, label1, label2, label3, label4, label5, node_counts, transform=None,
                 pre_transform=None):
        self.label1 = label1
        self.label2 = label2
        self.label3 = label3
        self.label4 = label4
        self.label5 = label5
        self.node_counts = node_counts
        self.var_size = 5
        self.src_data = pd.read_csv('../PublicSrcdata/merge_data_mit_5labels.csv')
        self.break_num = 100000
        # 数据的下载和处理过程在父类中调用实现
        super(GraphMitDataset, self).__init__(root, transform, pre_transform)
        # 加载数据
        self.data, self.slices = torch.load(self.processed_paths[0])

    # 将函数修饰为类属性
    @property
    def raw_file_names(self):
        return ['file_1', 'file_2']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def node_features(self, floor_num: int, direction: [], pkt_size: [], same_floor_dict: {}):
        """添加节点特征
        同层个数，上层突发个数，下层突发个数，上/同层包个数比，下/同层包个数比
        包长度，同层包总长度，上层包总长度，下层包总长度
        包长度同层占比，字节/同层包个数（平均包字节数）
        """
        # 每张图的特征向量
        pic_features_vec = []
        # 每个节点的特征向量
        node_vec = []
        # 同层包个数，上层包个数，下层包个数，同层字节数，上层字节数，下层字节数
        same_floor_num = 0
        last_floor_num = 0
        next_floor_num = 0
        same_floor_size = 0
        last_floor_size = 0
        next_floor_size = 0

        for floor in range(0, floor_num + 1):
            if floor == 0:
                # 获取该层每个节点的index
                for node_same in same_floor_dict[floor]:
                    same_floor_size += int(pkt_size[node_same])  # 计算该层字节总数
                if floor_num != 0:
                    for node_next in same_floor_dict[floor + 1]:
                        next_floor_size += int(pkt_size[node_next])  # 计算下层字节总数
                    next_floor_num = len(same_floor_dict[floor + 1])  # 下层节点数
                same_floor_num = len(same_floor_dict[floor])  # 计算该层总节点数

                '''添加节点特征
                同层个数，上层突发个数，下层突发个数，上/同层包个数比，下/同层包个数比
                包长度，同层包总长度，上层包总长度，下层包总长度
                包长度同层占比，字节/同层包个数（平均包字节数）
                '''
                for node in same_floor_dict[floor]:
                    if direction[node] == '1':
                        node_vec.append(int(pkt_size[node]))
                    elif direction[node] == '0':
                        node_vec.append(0 - int(pkt_size[node]))
                    node_vec.append(same_floor_num)
                    node_vec.append(last_floor_num)
                    node_vec.append(next_floor_num)
                    node_vec.append(last_floor_num / same_floor_num)
                    node_vec.append(next_floor_num / same_floor_num)
                    node_vec.append(same_floor_size)
                    node_vec.append(last_floor_size)
                    node_vec.append(next_floor_size)
                    node_vec.append(int(pkt_size[node]) / same_floor_size)
                    node_vec.append(same_floor_size / same_floor_num)

                    # 0329 add feature
                    if last_floor_size  == 0:
                        node_vec.append(0)
                    else:
                        node_vec.append(same_floor_size / last_floor_size)
                    if next_floor_size == 0:
                        node_vec.append(0)
                    else:
                        node_vec.append(same_floor_size / next_floor_size)

                    # 添加到特征矩阵中
                    pic_features_vec.append(copy.deepcopy(node_vec))
                    node_vec.clear()
            elif floor == floor_num:
                # 获取该层每个节点的index
                for node_same in same_floor_dict[floor]:
                    same_floor_size += int(pkt_size[node_same])  # 计算该层字节总数
                for node_last in same_floor_dict[floor - 1]:
                    last_floor_size += int(pkt_size[node_last])  # 计算上层字节总数

                same_floor_num = len(same_floor_dict[floor])  # 计算该层总节点数
                last_floor_num = len(same_floor_dict[floor - 1])  # 上层节点数
                '''添加节点特征
                同层个数，上层突发个数，下层突发个数，上/同层包个数比，下/同层包个数比
                包长度，同层包总长度，上层包总长度，下层包总长度
                包长度同层占比，字节/同层包个数（平均包字节数）
                '''
                for node in same_floor_dict[floor]:
                    if direction[node] == '1':
                        node_vec.append(int(pkt_size[node]))
                    elif direction[node] == '0':
                        node_vec.append(0 - int(pkt_size[node]))
                    node_vec.append(same_floor_num)
                    node_vec.append(last_floor_num)
                    node_vec.append(next_floor_num)
                    node_vec.append(last_floor_num / same_floor_num)
                    node_vec.append(next_floor_num / same_floor_num)
                    node_vec.append(same_floor_size)
                    node_vec.append(last_floor_size)
                    node_vec.append(next_floor_size)
                    node_vec.append(int(pkt_size[node]) / same_floor_size)
                    node_vec.append(same_floor_size / same_floor_num)

                    # 0329 add feature
                    if last_floor_size  == 0:
                        node_vec.append(0)
                    else:
                        node_vec.append(same_floor_size / last_floor_size)
                    if next_floor_size == 0:
                        node_vec.append(0)
                    else:
                        node_vec.append(same_floor_size / next_floor_size)

                    # 添加到特征矩阵中
                    pic_features_vec.append(copy.deepcopy(node_vec))
                    node_vec.clear()
            else:
                # 获取该层每个节点的index
                for node_same in same_floor_dict[floor]:
                    same_floor_size += int(pkt_size[node_same])  # 计算该层字节总数
                for node_last in same_floor_dict[floor - 1]:
                    last_floor_size += int(pkt_size[node_last])  # 计算上层字节总数
                for node_next in same_floor_dict[floor + 1]:
                    next_floor_size += int(pkt_size[node_next])  # 计算下层字节总数

                same_floor_num = len(same_floor_dict[floor])  # 计算该层总节点数
                last_floor_num = len(same_floor_dict[floor - 1])  # 上层节点数
                next_floor_num = len(same_floor_dict[floor + 1])  # 下层节点数
                '''添加节点特征
                包长度，
                同层个数，上层突发个数，下层突发个数，上/同层包个数比，下/同层包个数比
                同层包总长度，上层包总长度，下层包总长度
                包长度同层占比，字节/同层包个数（平均包字节数）
                '''
                for node in same_floor_dict[floor]:
                    if direction[node] == '1':
                        node_vec.append(int(pkt_size[node]))
                    elif direction[node] == '0':
                        node_vec.append(0 - int(pkt_size[node]))
                    node_vec.append(same_floor_num)
                    node_vec.append(last_floor_num)
                    node_vec.append(next_floor_num)
                    node_vec.append(last_floor_num / same_floor_num)
                    node_vec.append(next_floor_num / same_floor_num)
                    node_vec.append(same_floor_size)
                    node_vec.append(last_floor_size)
                    node_vec.append(next_floor_size)
                    node_vec.append(int(pkt_size[node]) / same_floor_size)
                    node_vec.append(same_floor_size / same_floor_num)

                    # 0329 add feature
                    if last_floor_size  == 0:
                        node_vec.append(0)
                    else:
                        node_vec.append(same_floor_size / last_floor_size)
                    if next_floor_size == 0:
                        node_vec.append(0)
                    else:
                        node_vec.append(same_floor_size / next_floor_size)

                    # 添加到特征矩阵中
                    pic_features_vec.append(copy.deepcopy(node_vec))
                    node_vec.clear()
            # 一层循环完后全部归零
            same_floor_num = 0
            last_floor_num = 0
            next_floor_num = 0
            same_floor_size = 0
            last_floor_size = 0
            next_floor_size = 0

        return pic_features_vec

    def generate_new_graph(self, direction_list: [], pkt_size_list: [], label):
        """生成无向图"""
        dir_length = len(direction_list)
        # 生成邻接矩阵存边
        neighbor_matrix = np.zeros((dir_length, dir_length))
        # 突发转换标志位
        change_flag = direction_list[0]
        # 同层字典
        same_floor_dict = {}
        # 层数
        floor_num = 0
        #0329 边属性
        edge_attr_list = []
        for index in range(0, dir_length):
            # index是节点序号，这个循环等于遍历所有节点
            if index == 0:
                same_floor_dict.update({floor_num: [index]})
                continue
            if direction_list[index] == change_flag:
                same_floor_dict[floor_num].append(index)
            else:
                # 不等于前面一个的标志位说明转换了，层数+1，且标志位需要转换
                floor_num += 1
                change_flag = direction_list[index]
                same_floor_dict.update({floor_num: [index]})

        # 遍历所有层，添加边
        for i in range(0, floor_num + 1):
            # 遍历同层节点
            for seq_index, seq in enumerate(same_floor_dict[i]):
                if seq_index == 0:
                    continue
                neighbor_matrix[seq][seq - 1] = 1
                neighbor_matrix[seq - 1][seq] = 1


            if i == 0:
                continue
            for last_seq in same_floor_dict[i - 1]:
                for this_seq in same_floor_dict[i]:
                    neighbor_matrix[this_seq][last_seq] = 1
                    neighbor_matrix[last_seq][this_seq] = 1


        edge_index = np.argwhere(neighbor_matrix == 1).tolist()
        Y = [int(label)]
        return edge_index, Y, self.node_features(floor_num, direction_list, pkt_size_list, same_floor_dict)

    def process(self):
        data_list = []  # 存放所有图的列表
        # --------以下为要建立图实例的地方--------
        data_src = self.src_data
        label1_data = data_src.loc[data_src['label'] == 1]
        label2_data = data_src.loc[data_src['label'] == 2]
        label3_data = data_src.loc[data_src['label'] == 3]
        label4_data = data_src.loc[data_src['label'] == 4]
        label5_data = data_src.loc[data_src['label'] == 5]

        # 数据包方向
        label1_pkt_direction = label1_data['udps.bi_flow_pkt_direction'].tolist()
        label2_pkt_direction = label2_data['udps.bi_flow_pkt_direction'].tolist()
        label3_pkt_direction = label3_data['udps.bi_flow_pkt_direction'].tolist()
        label4_pkt_direction = label4_data['udps.bi_flow_pkt_direction'].tolist()
        label5_pkt_direction = label5_data['udps.bi_flow_pkt_direction'].tolist()

        # 数据包大小
        label1_pkt_size = label1_data['udps.bi_pkt_size'].tolist()
        label2_pkt_size = label2_data['udps.bi_pkt_size'].tolist()
        label3_pkt_size = label3_data['udps.bi_pkt_size'].tolist()
        label4_pkt_size = label4_data['udps.bi_pkt_size'].tolist()
        label5_pkt_size = label5_data['udps.bi_pkt_size'].tolist()

        print('label1 length:' + str(len(label1_pkt_direction)))
        print('label2 length:' + str(len(label2_pkt_direction)))
        print('label3 length:' + str(len(label3_pkt_direction)))
        print('label4 length:' + str(len(label4_pkt_direction)))
        print('label5 length:' + str(len(label5_pkt_direction)))

        del label1_data
        del label2_data
        del label3_data
        del label4_data
        del label5_data

        index_cnt = 0
        for index in range(0, len(label1_pkt_direction)):
            if label1_pkt_direction[index] != label1_pkt_direction[index]:
                continue  # 如果出现空值，则直接下一条数据
            temp_list = label1_pkt_direction[index].split(' ')
            pkt_size_temp = label1_pkt_size[index].split(' ')
            temp_list.pop()
            pkt_size_temp.pop()
            if len(temp_list) < self.var_size or (np.array(temp_list[0:self.node_counts]) == '1').all() \
                    or (np.array(temp_list[0:self.node_counts]) == '0').all():
                continue
            temp_list = temp_list[0:self.node_counts]
            pkt_size_temp = pkt_size_temp[0:self.node_counts]

            edge_index, Y, x= self.generate_new_graph(temp_list, pkt_size_list=pkt_size_temp, label=self.label1)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            Y = torch.tensor(Y, dtype=torch.long)
            x = torch.tensor(x, dtype=torch.float)

            data = Data(x=x, edge_index=edge_index.t().contiguous(), y=Y)
            data_list.append(data)
            index_cnt += 1
            if self.break_num == index_cnt:
                break
        del label1_pkt_direction
        l1 = len(data_list)
        print('label1有' + str(l1) + '条数据')

        index_cnt2 = 0
        for index2 in range(0, len(label2_pkt_direction)):
            if label2_pkt_direction[index2] != label2_pkt_direction[index2]:
                continue  # 如果出现空值，则直接下一条数据
            temp_list = label2_pkt_direction[index2].split(' ')
            pkt_size_temp = label2_pkt_size[index2].split(' ')
            temp_list.pop()
            pkt_size_temp.pop()
            if len(temp_list) < self.var_size or (np.array(temp_list[0:self.node_counts]) == '1').all() \
                    or (np.array(temp_list[0:self.node_counts]) == '0').all():
                continue
            temp_list = temp_list[0:self.node_counts]
            pkt_size_temp = pkt_size_temp[0:self.node_counts]

            edge_index, Y, x= self.generate_new_graph(temp_list, pkt_size_list=pkt_size_temp, label=self.label2)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            Y = torch.tensor(Y, dtype=torch.long)
            x = torch.tensor(x, dtype=torch.float)


            data = Data(x=x,edge_index=edge_index.t().contiguous(), y=Y)
            data_list.append(data)
            index_cnt2 += 1
            # if self.break_num == index_cnt2:
            #     break
        del label2_pkt_direction
        l2 = len(data_list)
        print('label2有' + str(l2 - l1) + '条数据')

        index_cnt3 = 0
        for index3 in range(0, len(label3_pkt_direction)):
            if label3_pkt_direction[index3] != label3_pkt_direction[index3]:
                continue  # 如果出现空值，则直接下一条数据
            temp_list = label3_pkt_direction[index3].split(' ')
            pkt_size_temp = label3_pkt_size[index3].split(' ')
            temp_list.pop()
            pkt_size_temp.pop()
            if len(temp_list) < self.var_size or (np.array(temp_list[0:self.node_counts]) == '1').all() \
                    or (np.array(temp_list[0:self.node_counts]) == '0').all():
                continue
            temp_list = temp_list[0:self.node_counts]
            pkt_size_temp = pkt_size_temp[0:self.node_counts]

            edge_index, Y, x= self.generate_new_graph(temp_list, pkt_size_list=pkt_size_temp, label=self.label3)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            Y = torch.tensor(Y, dtype=torch.long)
            x = torch.tensor(x, dtype=torch.float)


            data = Data(x=x,edge_index=edge_index.t().contiguous(), y=Y)
            data_list.append(data)
            index_cnt3 += 1
            if self.break_num == index_cnt3:
                break
        del label3_pkt_direction
        l3 = len(data_list)
        print('label3有' + str(l3 - l2) + '条数据')

        index_cnt4 = 0
        for index4 in range(0, len(label4_pkt_direction)):
            if label4_pkt_direction[index4] != label4_pkt_direction[index4]:
                continue  # 如果出现空值，则直接下一条数据
            temp_list = label4_pkt_direction[index4].split(' ')
            pkt_size_temp = label4_pkt_size[index4].split(' ')
            temp_list.pop()
            pkt_size_temp.pop()
            if len(temp_list) < self.var_size or (np.array(temp_list[0:self.node_counts]) == '1').all() \
                    or (np.array(temp_list[0:self.node_counts]) == '0').all():
                continue
            temp_list = temp_list[0:self.node_counts]
            pkt_size_temp = pkt_size_temp[0:self.node_counts]

            edge_index, Y, x = self.generate_new_graph(temp_list, pkt_size_list=pkt_size_temp, label=self.label4)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            Y = torch.tensor(Y, dtype=torch.long)
            x = torch.tensor(x, dtype=torch.float)

            data = Data(x=x,edge_index=edge_index.t().contiguous(), y=Y)
            data_list.append(data)
            index_cnt4 += 1
            if self.break_num == index_cnt4:
                break
        del label4_pkt_direction
        l4 = len(data_list)
        print('label4有' + str(l4 - l3) + '条数据')

        index_cnt5 = 0
        for index5 in range(0,len(label5_pkt_direction)):
            if label5_pkt_direction[index5] != label5_pkt_direction[index5]:
                continue  # 如果出现空值，则直接下一条数据
            temp_list = label5_pkt_direction[index5].split(' ')
            pkt_size_temp = label5_pkt_size[index5].split(' ')
            temp_list.pop()
            pkt_size_temp.pop()
            # if len(temp_list) < self.var_size or (np.array(temp_list[0:self.node_counts]) == '1').all() \
            #         or (np.array(temp_list[0:self.node_counts]) == '0').all():
            if len(temp_list) < self.var_size:
                continue
            temp_list = temp_list[0:self.node_counts]
            pkt_size_temp = pkt_size_temp[0:self.node_counts]
            # self.node_counts = len(temp_list)

            edge_index, Y, x= self.generate_new_graph(temp_list, pkt_size_list=pkt_size_temp, label=self.label5)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            Y = torch.tensor(Y, dtype=torch.long)
            x = torch.tensor(x, dtype=torch.float)

            data = Data(x=x, edge_index=edge_index.t().contiguous(), y=Y)
            data_list.append(data)
            index_cnt5 += 1
            if self.break_num == index_cnt5:
                break
        del label5_pkt_direction
        l5 = len(data_list)
        print('label5有' + str(l5 - l4) + '条数据')

        random.shuffle(data_list)
        print('共计' + str(len(data_list)) + '条数据')
        # --------以上为要建立图实例的地方--------
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_filter is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        # 这里的save方式以及路径需要对应构造函数中的load操作
        torch.save((data, slices), self.processed_paths[0])

if __name__ == '__main__':
    pass
