"""
@Created Time : 2022/11/01
@Author  : LiYao
@FileName: DrawPicture.py
@Description:做事前分析以及统计图的绘制
@Modified:
    :First modified
    :Modified content:
"""

from pylab import *
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from itertools import product
import torch
import torch.nn as nn
import os
import numpy as np

import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score


class Graph_Matrix(object):
    """
    Adjacency Matrix
    """

    def __init__(self, vertices=[], matrix=[]):
        """

        :param vertices:a dict with vertex id and index of matrix , such as {vertex:index}
        :param matrix: a matrix
        """
        self.matrix = matrix
        self.edges_dict = {}  # {(tail, head):weight}
        self.edges_array = []  # (tail, head, weight)
        self.vertices = vertices
        self.num_edges = 0

        # if provide adjacency matrix then create the edges list
        if len(matrix) > 0:
            if len(vertices) != len(matrix):
                raise IndexError
            self.edges = self.getAllEdges()
            self.num_edges = len(self.edges)

        # if do not provide a adjacency matrix, but provide the vertices list, build a matrix with 0
        elif len(vertices) > 0:
            self.matrix = [[0 for col in range(len(vertices))] for row in range(len(vertices))]

        self.num_vertices = len(self.matrix)

    def isOutRange(self, x):
        try:
            if x >= self.num_vertices or x <= 0:
                raise IndexError
        except IndexError:
            print("节点下标出界")

    def isEmpty(self):
        if self.num_vertices == 0:
            self.num_vertices = len(self.matrix)
        return self.num_vertices == 0

    def add_vertex(self, key):
        if key not in self.vertices:
            self.vertices[key] = len(self.vertices) + 1

        # add a vertex mean add a row and a column
        # add a column for every row
        for i in range(self.getVerticesNumbers()):
            self.matrix[i].append(0)

        self.num_vertices += 1

        nRow = [0] * self.num_vertices
        self.matrix.append(nRow)

    def getVertex(self, key):
        pass

    def add_edges_from_list(self, edges_list):  # edges_list : [(tail, head, weight),()]
        for i in range(len(edges_list)):
            self.add_edge(edges_list[i][0], edges_list[i][1], edges_list[i][2], )

    def add_edge(self, tail, head, cost=0):
        # if self.vertices.index(tail) >= 0:
        #   self.addVertex(tail)
        if tail not in self.vertices:
            self.add_vertex(tail)
        # if self.vertices.index(head) >= 0:
        #   self.addVertex(head)
        if head not in self.vertices:
            self.add_vertex(head)

        # for directory matrix
        self.matrix[self.vertices.index(tail)][self.vertices.index(head)] = cost
        # for non-directory matrix
        # self.matrix[self.vertices.index(fromV)][self.vertices.index(toV)] = \
        #   self.matrix[self.vertices.index(toV)][self.vertices.index(fromV)] = cost

        self.edges_dict[(tail, head)] = cost
        self.edges_array.append((tail, head, cost))
        self.num_edges = len(self.edges_dict)

    def getEdges(self, V):
        pass

    def getVerticesNumbers(self):
        if self.num_vertices == 0:
            self.num_vertices = len(self.matrix)
        return self.num_vertices

    def getAllVertices(self):
        return self.vertices

    def getAllEdges(self):
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                if 0 < self.matrix[i][j] < float('inf'):
                    self.edges_dict[self.vertices[i], self.vertices[j]] = self.matrix[i][j]
                    self.edges_array.append([self.vertices[i], self.vertices[j], self.matrix[i][j]])

        return self.edges_array

    def __repr__(self):
        return str(''.join(str(i) for i in self.matrix))

    def to_do_vertex(self, i):
        print('vertex: %s' % (self.vertices[i]))

    def to_do_edge(self, w, k):
        print('edge tail: %s, edge head: %s, weight: %s' % (self.vertices[w], self.vertices[k], str(self.matrix[w][k])))


class DrawPic(object):
    def __init__(self):
        self.file_path = '../PublicSrcdata/merge_data_mit_5labels.csv'
        self.data_src = pd.read_csv(self.file_path)

    def before_10pkt_up_down_liner(self):
        """
        所有流前10/20个包的上下行个数
        :return:
        """
        data_src = self.data_src
        label1_data = data_src.loc[data_src['label'] == (1)]
        label2_data = data_src.loc[data_src['label'] == (2)]
        # label3_data = data_src.loc[data_src['label'] == (3)]
        # label10_data = data_src.loc[data_src['label'] == (10)]

        # # rdp前10包方向统计
        label1_10pkt_stat_list_pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 正向统计
        label1_10pkt_stat_list_neg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 负向统计
        label1_10pkt_stat_list_pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 正向统计
        label1_10pkt_stat_list_neg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 负向统计
        label1_pkt_direction = label1_data['udps.bi_flow_pkt_direction'].tolist()
        for value in label1_pkt_direction:
            if value != value:
                continue  # 如果出现空值，则直接下一条数据
            temp_list = value.split(' ')
            if len(temp_list) <= 20:
                pass
            else:
                for index in range(0, 20):
                    if int(temp_list[index]) == 1:
                        label1_10pkt_stat_list_pos[index] += 1
                    else:
                        label1_10pkt_stat_list_neg[index] += 1
        # voip前10包方向统计
        label2_10pkt_stat_list_pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 正向统计
        label2_10pkt_stat_list_neg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 负向统计
        label2_10pkt_stat_list_pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 正向统计
        label2_10pkt_stat_list_neg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 负向统计
        label2_pkt_direction = label2_data['udps.bi_flow_pkt_direction'].tolist()
        for value in label2_pkt_direction:
            if value != value:
                continue  # 如果出现空值，则直接下一条数据
            temp_list = value.split(' ')
            if len(temp_list) <= 20:
                pass
            else:
                for index in range(0, 20):
                    if int(temp_list[index]) == 1:
                        label2_10pkt_stat_list_pos[index] += 1
                    else:
                        label2_10pkt_stat_list_neg[index] += 1
        # # youtube前10包方向统计
        # label3_10pkt_stat_list_pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]#正向统计
        # label3_10pkt_stat_list_neg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]#负向统计
        # label3_10pkt_stat_list_pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0]#正向统计
        # label3_10pkt_stat_list_neg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0]#负向统计
        # label3_pkt_direction = label3_data['udps.bi_flow_pkt_direction'].tolist()
        # for value in label3_pkt_direction:
        #     if value != value:
        #         continue  # 如果出现空值，则直接下一条数据
        #     temp_list = value.split(' ')
        #     if len(temp_list) <= 20:
        #         pass
        #     else:
        #         for index in range(0, 20):
        #             if int(temp_list[index]) == 1:
        #                 label3_10pkt_stat_list_pos[index] += 1
        #             else:
        #                 label3_10pkt_stat_list_neg[index] += 1

        # ssh前10包方向统计
        # label10_10pkt_stat_list_pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]#正向统计
        # label10_10pkt_stat_list_neg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]#负向统计
        # label10_10pkt_stat_list_pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0]#正向统计
        # label10_10pkt_stat_list_neg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0]#负向统计
        # label10_pkt_direction = label10_data['udps.bi_flow_pkt_direction'].tolist()
        # for value in label10_pkt_direction:
        #     if value != value:
        #         continue  # 如果出现空值，则直接下一条数据
        #     temp_list = value.split(' ')
        #     if len(temp_list) <= 20:
        #         pass
        #     else:
        #         for index in range(0, 20):
        #             if int(temp_list[index]) == 1:
        #                 label10_10pkt_stat_list_pos[index] += 1
        #             else:
        #                 label10_10pkt_stat_list_neg[index] += 1

        x_axis_data = []
        for x in range(1, 21):
            x_axis_data.append(x)
        # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
        # plt.plot(x_axis_data, label1_10pkt_stat_list_pos, 'o-', color='r', alpha=0.8, linewidth=1, label='rdp+')
        # plt.plot(x_axis_data, label1_10pkt_stat_list_neg, '+:', color='m', alpha=0.8, linewidth=1, label='rdp-')

        # plt.plot(x_axis_data, label2_10pkt_stat_list_pos, 'o-', color='r', alpha=0.8, linewidth=1, label='voip+')
        # plt.plot(x_axis_data, label2_10pkt_stat_list_neg, '+:', color='m', alpha=0.8, linewidth=1, label='voip-')

        # plt.plot(x_axis_data, label3_10pkt_stat_list_pos, 'o-', color='r', alpha=0.8, linewidth=1, label='youtube+')
        # plt.plot(x_axis_data, label3_10pkt_stat_list_neg, '+:', color='m', alpha=0.8, linewidth=1, label='youtube-')
        plt.plot(x_axis_data, label2_10pkt_stat_list_pos, 'o-', color='r', alpha=0.8, linewidth=1, label='ssh+')
        plt.plot(x_axis_data, label2_10pkt_stat_list_neg, '+:', color='m', alpha=0.8, linewidth=1, label='ssh-')

        # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
        plt.legend(loc="upper right")
        plt.xlabel('pkt')
        plt.ylabel('number')

        plt.show()
        # plt.savefig('demo.jpg')  # 保存该图片

    def draw_behavior_picture(self):
        """画出来前十个包构成的行为交互图"""
        data_src = self.data_src
        # label1_data = data_src.loc[data_src['label'] == (1)]
        # label2_data = data_src.loc[data_src['label'] == (2)]
        # label3_data = data_src.loc[data_src['label'] == (3)]
        # label4_data = data_src.loc[data_src['label'] == (4)]
        label5_data = data_src.loc[data_src['label'] == (5)]

        label1_pkt_direction = label5_data['udps.bi_flow_pkt_direction'].tolist()
        var_size = 10
        nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        cnt = 0
        for index_all, value in enumerate(label1_pkt_direction):
            if value != value:
                continue  # 如果出现空值，则直接下一条数据
            value = value.split(' ')
            value.pop()
            value = value[0:10]
            if len(value) < var_size or (np.array(value) == '1').all() \
                    or (np.array(value) == '0').all():
                continue
            print(str(cnt) + ': ' + str(value))
            a = np.zeros((var_size, var_size))
            first_flag = 0
            last_flag = -1
            for index in range(0, var_size):
                if index == 0:
                    continue
                if value[index] == value[index - 1]:
                    a[index - 1][index] = 1
                    a[index][index - 1] = 1

                else:
                    a[first_flag][index] = 1
                    a[index][first_flag] = 1

                    first_flag = index
                    if last_flag == -1:
                        last_flag = index - 1
                    else:
                        temp_flag = a[last_flag][index - 1]
                        a[last_flag][index - 1] = 1
                        a[index - 1][last_flag] = 1

                        last_flag = index - 1
                if index == var_size - 1:
                    temp_flag = a[last_flag][index]
                    a[last_flag][index] = 1
                    a[index][last_flag] = 1

            this_graph = Graph_Matrix(nodes, a)
            # print(this_graph)
            G = nx.Graph()  # 建立一个空的无向图G
            for node in this_graph.vertices:
                G.add_node(str(node))
            for edge in this_graph.edges:
                G.add_edge(str(edge[0]), str(edge[1]))
            # print("nodes:", G.nodes())  # 输出全部的节点： [1, 2, 3]
            # print("edges:", G.edges())  # 输出全部的边：[(2, 3)]
            # print("number of edges:", G.number_of_edges())  # 输出边的数量：1
            nx.draw(G, with_labels=True)
            plt.show()
            plt.savefig("undirected_graph" + str(cnt) + ".png")
            cnt += 1
            plt.close()

    def generate_new_graph(self):
        data_src = self.data_src
        # label1_data = data_src.loc[data_src['label'] == (1)]
        # label2_data = data_src.loc[data_src['label'] == (2)]
        # label3_data = data_src.loc[data_src['label'] == (3)]
        # label4_data = data_src.loc[data_src['label'] == (4)]
        label5_data = data_src.loc[data_src['label'] == (5)]

        label1_pkt_direction = label5_data['udps.bi_flow_pkt_direction'].tolist()
        var_size = 10
        nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        cnt = 0
        for index_all, direction_list in enumerate(label1_pkt_direction):
            if direction_list != direction_list:
                continue  # 如果出现空值，则直接下一条数据
            direction_list = direction_list.split(' ')
            direction_list.pop()
            direction_list = direction_list[0:10]
            if len(direction_list) < 10 or (np.array(direction_list) == '1').all() \
                    or (np.array(direction_list) == '0').all():
                continue
            print(str(cnt) + ': ' + str(direction_list))

            dir_length = len(direction_list)
            # 生成邻接矩阵存边
            neighbor_matrix = np.zeros((dir_length, dir_length))
            # 突发转换标志位
            change_flag = direction_list[0]
            # 同层字典
            same_floor_dict = {}
            # 层数
            floor_num = 0
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

            this_graph = Graph_Matrix(nodes, neighbor_matrix)
            # print(this_graph)
            G = nx.Graph()  # 建立一个空的无向图G
            for node in this_graph.vertices:
                G.add_node(str(node))
            for edge in this_graph.edges:
                G.add_edge(str(edge[0]), str(edge[1]))
            # print("nodes:", G.nodes())  # 输出全部的节点： [1, 2, 3]
            # print("edges:", G.edges())  # 输出全部的边：[(2, 3)]
            # print("number of edges:", G.number_of_edges())  # 输出边的数量：1
            nx.draw(G, with_labels=True)
            # plt.savefig("./label5pic/undirected_graph"+str(cnt)+".png")
            # plt.close()
            plt.show()

    # 显示混淆矩阵
    def plot_confuse(self, y_true, y_pred, path):

        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        plt.figure()
        # 指定分类类别
        classes = range(np.max(y_true) + 1)
        title = 'Confusion matrix'
        # 混淆矩阵颜色风格
        # cmap = plt.cm.jet
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.imshow(cm, cmap=plt.cm.Greens)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        thresh = cm.max() / 2.
        # 按照行和列填写百分比数据
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            ll = '{:.2f}'.format(cm[i, j])
            if ll == '1.00':
                pass
            plt.text(j, i, ll, horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(path)
        plt.show()
    def ROC(self, num_class,label_list,score_list,path):
        score_array = np.array(score_list)
        # 将label转换成onehot形式
        label_tensor = torch.tensor(label_list)
        label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
        label_onehot = torch.zeros(label_tensor.shape[0], num_class)
        label_onehot.scatter_(dim=1, index=label_tensor, value=1)
        label_onehot = np.array(label_onehot)

        print("score_array:", score_array.shape)  # (batchsize, classnum)
        print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])

        # 调用sklearn库，计算每个类别对应的fpr和tpr
        fpr_dict = dict()
        tpr_dict = dict()
        roc_auc_dict = dict()
        for i in range(num_class):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        # micro
        fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
        roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

        # macro
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_class):
            mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
        # Finally average it and compute AUC
        mean_tpr /= num_class
        fpr_dict["macro"] = all_fpr
        tpr_dict["macro"] = mean_tpr
        roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

        # 绘制所有类别平均的roc曲线
        plt.figure()
        lw = 2
        # plt.plot(fpr_dict["micro"], tpr_dict["micro"],
        #          label='micro-average ROC curve (area = {0:0.2f})'
        #                ''.format(roc_auc_dict["micro"]),
        #          color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr_dict["macro"], tpr_dict["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc_dict["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(num_class), colors):
            plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc_dict[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.savefig(path)
        plt.show()

    def PR(self,num_class,label_list,score_list,path):
        score_array = np.array(score_list)
        # 将label转换成onehot形式
        label_tensor = torch.tensor(label_list)
        label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
        label_onehot = torch.zeros(label_tensor.shape[0], num_class)
        label_onehot.scatter_(dim=1, index=label_tensor, value=1)
        label_onehot = np.array(label_onehot)
        print("score_array:", score_array.shape)  # (batchsize, classnum) softmax
        print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum]) onehot

        # 调用sklearn库，计算每个类别对应的precision和recall
        precision_dict = dict()
        recall_dict = dict()
        average_precision_dict = dict()
        for i in range(num_class):
            precision_dict[i], recall_dict[i], _ = precision_recall_curve(label_onehot[:, i], score_array[:, i])
            average_precision_dict[i] = average_precision_score(label_onehot[:, i], score_array[:, i])
            print(precision_dict[i].shape, recall_dict[i].shape, average_precision_dict[i])

        # micro
        precision_dict["micro"], recall_dict["micro"], _ = precision_recall_curve(label_onehot.ravel(),
                                                                                  score_array.ravel())
        average_precision_dict["micro"] = average_precision_score(label_onehot, score_array, average="micro")
        print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(
            average_precision_dict["micro"]))

        # 绘制所有类别平均的pr曲线
        plt.figure()
        plt.step(recall_dict['micro'], precision_dict['micro'], where='post')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(
            'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
                .format(average_precision_dict["micro"]))
        plt.savefig(path)
        plt.show()



    # 显示混淆矩阵
    def plot_confuse_number(self, y_true, y_pred, path):

        confusion = confusion_matrix(y_true=y_true, y_pred=y_pred)
        # 颜色风格为绿。。。。
        plt.imshow(confusion, cmap=plt.cm.Greens)
        # ticks 坐标轴的坐标点
        # label 坐标轴标签说明
        classes = range(np.max(y_true) + 1)
        indices = range(len(confusion))
        # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
        plt.xticks(indices, classes)
        plt.yticks(indices, classes)
        plt.colorbar()
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion matrix')

        # plt.rcParams两行是用于解决标签不能显示汉字的问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 显示数据
        for first_index in range(len(confusion)):  # 第几行
            for second_index in range(len(confusion[first_index])):  # 第几列
                plt.text(first_index, second_index, confusion[first_index][second_index])
        plt.savefig(path)
        # 显示
        plt.show()


def plot_confuse_normalization():
    """1230用来规整画出来的混淆矩阵图"""
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 14.5,
            }
    # 1dcnn
    cm = [[0.95, 0.02, 0, 0, 0.03], [0.08, 0.85, 0.06, 0.00, 0.01],
          [0, 0.03, 0.95, 0, 0.02], [0.01, 0, 0.99, 0, 0]
        , [0.04, 0.05, 0, 0, 0.91]]
    # MAJOR exp
    # cm = [[0.97,0.01,0,0,0.02], [0.01,0.96,0,0,0.03],
    #       [0.01,0,0.90,0.08,0.01], [0,0,0.11,0.89,0]
    #     , [0.02,0.05,0,0,0.93]]
    # two feature exp
    # cm = [[0.88,0.03,0.01,0,0.08], [0.03,0.84,0.01,0,0.12],
    #       [0,0,0.88,0.09,0.03], [0,0.01,0.15,0.83,0.01]
    #     , [0.05,0.05,0.02,0.01,0.87]]
    # dapp exp
    # cm = [[0.86,0.06,0.01,0.03,0.04], [0.03,0.93,0.01,0.01,0.02],
    #       [0.01,0.01,0.86,0.10,0.02], [0.07,0.02,0.04,0.87,0]
    #     , [0.03,0.06,0.04,0.01,0.86]]
    cm = np.array(cm)
    plt.figure()

    title = ''
    # 混淆矩阵颜色风格
    # cmap = plt.cm.jet
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.imshow(cm, cmap=plt.cm.Blues)
    font_title = {'family': 'Times New Roman',
                  'weight': 'bold',
                  'size': 15,
                  }
    plt.title(title, font_title)
    plt.colorbar()
    tick_marks = [0, 1, 2, 3, 4]
    classes = [1, 2, 3, 4, 5]
    # classes=['Stream','Chat','C&C','File Transfer','VoIP']

    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.tick_params(which='major', length=1.2)
    plt.tick_params(which='major', width=1.5)

    thresh = cm.max() / 2.
    # 按照行和列填写百分比数据
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        ll = '{:.2f}'.format(cm[i, j])
        if ll == '1.00':
            ll = '0.99'
        plt.text(j, i, ll, horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.ylabel('True label', font)
    plt.xlabel('Predicted label', font)
    plt.savefig('./1dcnnexp1231.svg', bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    draw_object = DrawPic()
    # draw_object.before_10pkt_up_down_liner()
    # draw_object.draw_behavior_picture()
    # draw_object.generate_new_graph()
    plot_confuse_normalization()
