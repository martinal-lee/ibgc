"""
@Created Time : 2022/11/17
@Author  : LiYao
@FileName: GraphMainFunction.py
@Description:图神经网络的训练、执行等步骤的主函数
@Modified:
    :First modified
    :Modified content:
"""
import math
import random

from torch.utils.data import WeightedRandomSampler
from GraphNet import GraphNet, GATGraphNet,SAGEGraphNet
from torch_geometric.loader import DataLoader
from NewGraphDataset import GraphMitDataset,GraphMitDataset_bak
from sklearn.metrics import precision_score, classification_report
from DrawPicture import DrawPic
from sklearn.model_selection import  StratifiedKFold,KFold
import torch


class GraphMainClass(object):
    def __init__(self, data, num_classes, test_data):
        self.src_graph_model = SAGEGraphNet(num_classes).to(device)
        self.data = data  # 训练集
        self.test_data = test_data

    def train(self):
        optimizier = torch.optim.Adam(self.src_graph_model.parameters(), lr=0.0005, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        self.src_graph_model.train()

        train_batch_cnt = 0
        for data_batch in self.data:
            # print('train_batch_cnt '+str(train_batch_cnt))
            train_batch_cnt += 1
            target = data_batch.y.to(device)
            data_batch.to(device)
            out = self.src_graph_model(data_batch.x, data_batch.edge_index, data_batch.batch)  # 一次前向传播
            optimizier.zero_grad()
            loss = criterion(out, target)
            loss.backward()
            optimizier.step()
            #print(' loss {:.4f}'.format(loss.item()))
            #print('Epoch {:03d} loss {:.4f}'.format(epoch, loss.item()))

    def test(self, ):
        self.src_graph_model.eval()
        correct = 0
        test_batch_cnt = 0
        for data_batch in self.test_data:
            # print('test_batch_cnt '+str(test_batch_cnt))
            test_batch_cnt += 1
            data_batch.to(device)  # 批遍历测试集数据集。
            out = self.src_graph_model(data_batch.x, data_batch.edge_index, data_batch.batch)  # 一次前向传播
            pred = out.argmax(dim=1)  # 使用概率最高的类别
            correct += int((pred == data_batch.y.to(device)).sum())  # 检查真实标签

        return correct / len(self.test_data.dataset)

    def save_model(self, model_path):
        torch.save(self.src_graph_model, model_path)

    def predict(self, model_path):
        self.src_graph_model = torch.load(model_path)
        self.src_graph_model.eval()
        correct = 0
        true_list = []
        pred_list = []
        pred_score = []  # 概率向量，用于画pr roc
        for data_batch in self.test_data:
            data_batch.to(device)  # 批遍历测试集数据集。
            out = self.src_graph_model(data_batch.x, data_batch.edge_index, data_batch.batch)  # 一次前向传播
            pred_score.extend(out.tolist())
            pred = out.argmax(dim=1)  # 使用概率最高的类别
            correct += int((pred == data_batch.y.to(device)).sum())  # 检查真实标签
            true_list.append(data_batch.y.to(device).tolist())
            pred_list.append(pred.tolist())

        return correct / len(self.test_data.dataset), true_list, pred_list, pred_score

def pkt_num_select_exp():
    """选择数据包个数的实验"""
    label1 = 0
    label2 = 0
    label3 = 0
    label4 = 0
    label5 = 0
    pkt_list = [10,20,30,40,50,60,70,80]
    for label_value in range(1, 6):
        if label_value == 1:
            label1 = 0
            label2 = 1
            label3 = 1
            label4 = 1
            label5 = 1
        elif label_value == 2:
            label1 = 1
            label2 = 0
            label3 = 1
            label4 = 1
            label5 = 1
        elif label_value == 3:
            label1 = 1
            label2 = 1
            label3 = 0
            label4 = 1
            label5 = 1
        elif label_value == 4:
            label1 = 1
            label2 = 1
            label3 = 1
            label4 = 0
            label5 = 1
        elif label_value == 5:
            label1 = 1
            label2 = 1
            label3 = 1
            label4 = 1
            label5 = 0
        for pkt in pkt_list:
            true_all = []
            pred_all = []
            for ii in range(0, 10):
                dataset = GraphMitDataset(
                    root='./dataset_Major_input/pkt_select_pre_exp/packet_select_pre_exp_'+str(label_value)+'_label_'+str(pkt)+'nums',
                    label1=label1,
                    label2=label2,
                    label3=label3,
                    label4=label4, label5=label5,
                    node_counts=40)

                dataset = dataset.shuffle()
                dataset = dataset[0:500]
                train_size = int(0.7 * len(dataset))
                test_size = len(dataset) - train_size

                train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
                # labels_weights = []
                # labels_weights_test = []
                # for i in range(0, len(test_dataset)):
                #     if (dataset[test_dataset.indices[i]]).y[0] == 0:
                #         labels_weights_test.append(3854 / 1652)
                #     if (dataset[test_dataset.indices[i]]).y[0] == 1:
                #         labels_weights_test.append(3854 / 823)
                #     if (dataset[test_dataset.indices[i]]).y[0] == 2:
                #         labels_weights_test.append(3854 / 611)
                #     if (dataset[test_dataset.indices[i]]).y[0] == 3:
                #         labels_weights_test.append(3854 / 457)
                #     if (dataset[test_dataset.indices[i]]).y[0] == 4:
                #         labels_weights_test.append(3854 / 311)
                # for i in range(0, len(train_dataset)):
                #     if (dataset[train_dataset.indices[i]]).y[0] == 0:
                #         labels_weights.append(3854 / 1652)
                #     if (dataset[train_dataset.indices[i]]).y[0] == 1:
                #         labels_weights.append(3854 / 823)
                #     if (dataset[train_dataset.indices[i]]).y[0] == 2:
                #         labels_weights.append(3854 / 611)
                #     if (dataset[train_dataset.indices[i]]).y[0] == 3:
                #         labels_weights.append(3854 / 457)
                #     if (dataset[train_dataset.indices[i]]).y[0] == 4:
                #         labels_weights.append(3854 / 311)
                #
                # sampler_train = WeightedRandomSampler(labels_weights, num_samples=1500, replacement=True)
                # sampler_test = WeightedRandomSampler(labels_weights_test, num_samples=500, replacement=True)

                # train_set = DataLoader(train_dataset, batch_size=64, shuffle=False, sampler=sampler_train)
                # test_set = DataLoader(test_dataset, batch_size=64, shuffle=False, sampler=sampler_test)
                train_set = DataLoader(train_dataset, batch_size=64, shuffle=False, )
                test_set = DataLoader(test_dataset, batch_size=64, shuffle=False,)
                # 建立网络
                g_obj = GraphMainClass(train_set, 2, test_set)
                # 训练以及测试
                last_acc = 0
                for epoch in range(0, 50):
                    g_obj.train()
                    test_acc = g_obj.test()
                    if last_acc < test_acc:
                        last_acc = test_acc
                        g_obj.save_model('./model/test_model/pkt_select_exp')

                pred_acc, true_list, pred_list, pred_scr = g_obj.predict('./model/test_model/pkt_select_exp')
                true_list = [i for item in true_list for i in item]
                pred_list = [i for item in pred_list for i in item]
                true_all.extend(true_list)
                pred_all.extend(pred_list)
            print('**********************')
            print('第'+str(label_value)+'个标签，第'+str(pkt)+'数据包')
            print(classification_report(true_all, pred_all, digits=4))
            print("average micro accuracy is " + str(precision_score(true_all, pred_all, average='micro')))
            print("average macro accuracy is " + str(precision_score(true_all, pred_all, average='macro')))
            print('第'+str(label_value)+'个标签，第'+str(pkt)+'数据包')
            print('**********************')

def balance_data_sampler_multi_classify():
    """mit平衡数据集效果,sampler,直接多分类"""

    # 用于画多分类图
    true_all = []
    pred_all = []
    # 用于画roc和pr
    pred_scr_all = []
    for ii in range(0, 10):
        dataset = GraphMitDataset_bak(
            root='./dataset_Major_input/mit_multi_classify_test_add_feature_0612',
            label1=0,
            label2=1,
            label3=2,
            label4=3,
            label5=4, node_counts=20)

        dataset = dataset.shuffle()

        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        labels_weights = []
        labels_weights_test = []
        for i in range(0, len(test_dataset)):
            if (dataset[test_dataset.indices[i]]).y[0] == 0:
                labels_weights_test.append(3854 / 1652)
            if (dataset[test_dataset.indices[i]]).y[0] == 1:
                labels_weights_test.append(3854 / 823)
            if (dataset[test_dataset.indices[i]]).y[0] == 2:
                labels_weights_test.append(3854 / 611)
            if (dataset[test_dataset.indices[i]]).y[0] == 3:
                labels_weights_test.append(3854 / 457)
            if (dataset[test_dataset.indices[i]]).y[0] == 4:
                labels_weights_test.append(3854 / 311)
        for i in range(0, len(train_dataset)):
            if (dataset[train_dataset.indices[i]]).y[0] == 0:
                labels_weights.append(3854 / 1652)
            if (dataset[train_dataset.indices[i]]).y[0] == 1:
                labels_weights.append(3854 / 823)
            if (dataset[train_dataset.indices[i]]).y[0] == 2:
                labels_weights.append(3854 / 611)
            if (dataset[train_dataset.indices[i]]).y[0] == 3:
                labels_weights.append(3854 / 457)
            if (dataset[train_dataset.indices[i]]).y[0] == 4:
                labels_weights.append(3854 / 311)

        sampler_train = WeightedRandomSampler(labels_weights, num_samples=10000, replacement=True)
        sampler_test = WeightedRandomSampler(labels_weights_test, num_samples=5000, replacement=True)

        train_set = DataLoader(train_dataset, batch_size=128, shuffle=False, sampler=sampler_train)
        test_set = DataLoader(test_dataset, batch_size=128, shuffle=False, sampler=sampler_test)

        # 建立网络
        g_obj = GraphMainClass(train_set, 5, test_set)
        # 训练以及测试
        last_acc = 0
        for epoch in range(0, 50):
            g_obj.train()
            test_acc = g_obj.test()
            if last_acc < test_acc:
                last_acc = test_acc
                g_obj.save_model('./model/test_model/mit_multi_classify_test0329_add_feature_0612')

        pred_acc, true_list, pred_list, pred_scr = g_obj.predict('./model/test_model/mit_multi_classify_test0329_add_feature_0612')
        # print('test dataset acc is ' + str(pred_acc))

        true_list = [i for item in true_list for i in item]
        pred_list = [i for item in pred_list for i in item]
        print("第" + str(ii) + "次 micro accuracy is " + str(precision_score(true_list, pred_list, average='micro')))
        true_all.extend(true_list)
        pred_all.extend(pred_list)
        pred_scr_all.extend(pred_scr)

    print(classification_report(true_all, pred_all, digits=6))
    print("average micro accuracy is " + str(precision_score(true_all, pred_all, average='micro')))
    print("average macro accuracy is " + str(precision_score(true_all, pred_all, average='macro')))
    draw_obj.plot_confuse(true_all, pred_all, path='./picture_MajorExperiment/0612.png')
    # draw_obj.plot_confuse_number(true_all, pred_all, path='./picture_MajorExperiment/test0224num_5')
    # draw_obj.ROC(5, true_all, pred_scr_all, path='./picture_MajorExperiment/0301roc.png')
    # draw_obj.PR(5, true_all, pred_scr_all, path='./picture_MajorExperiment/0301pr.png')
    # draw_obj.plot_confuse(true_all, pred_all, path='./picture_MajorExperiment/test0316')
    # draw_obj.plot_confuse_number(true_all, pred_all, path='./picture_MajorExperiment/test0316num')
    # draw_obj.ROC(5, true_all, pred_scr_all, path='./picture_MajorExperiment/0316roc.png')
    # draw_obj.PR(5, true_all, pred_scr_all, path='./picture_MajorExperiment/0316pr.png')



if __name__ == '__main__':
    # cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    draw_obj = DrawPic()

    #pkt_num_select_exp()
    balance_data_sampler_multi_classify()
    #balance_data_sampler_multi_classify_tor()
    #balance_data_sampler_multi_classify_vpn()
    # balance_data_sampler_multi_classify_datacon()
