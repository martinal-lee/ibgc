"""
@Created Time : 2022/11/17
@Author  : LiYao
@FileName: GraphNet.py
@Description:生成图神经网络
@Modified:
    :First modified
    :Modified content:040312整理上传
"""
import torch
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv, GATConv,SAGEConv
import torch.nn.functional as F
from torch.nn import Linear


class GraphNet(torch.nn.Module):
    def __init__(self, num_classes):
        """

        :param num_node_features: 节点特征数
        :param num_classes: 节点类别数
        """
        super(GraphNet, self).__init__()
        # 第一卷积层
        self.conv1 = GCNConv(13, 16)
        # 第二卷积层
        self.conv2 = GCNConv(16, 64)
        self.conv3 = GCNConv(64, 128)
        self.lin = Linear(128, num_classes)

    def forward(self, x, edge_index,batch):
        # edge_index = self.conv1(edge_index)
        # edge_index = F.relu(edge_index)
        # edge_index = F.dropout(edge_index,training=self.training)
        # edge_index = self.conv2(edge_index)
        # edge_index = F.relu(edge_index)
        # edge_index = F.dropout(edge_index, training=self.training)
        # edge_index = F.softmax(edge_index,dim=1)
        # 1. 获得节点嵌入
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. 分类器
        x = F.dropout(x, p=0.025, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)

        return x

class SAGEGraphNet(torch.nn.Module):
    def __init__(self, num_classes):
        """

        :param num_node_features: 节点特征数
        :param num_classes: 节点类别数
        """
        super(SAGEGraphNet, self).__init__()
        # 第一卷积层
        self.conv1 = SAGEConv(13, 16,)
        # 第二卷积层
        self.conv2 = SAGEConv(16, 64,)
        self.conv3 = SAGEConv(64, 128,)
        self.lin = Linear(128, num_classes)

    def forward(self, x, edge_index,batch):
        # edge_index = self.conv1(edge_index)
        # edge_index = F.relu(edge_index)
        # edge_index = F.dropout(edge_index,training=self.training)
        # edge_index = self.conv2(edge_index)
        # edge_index = F.relu(edge_index)
        # edge_index = F.dropout(edge_index, training=self.training)
        # edge_index = F.softmax(edge_index,dim=1)
        # 1. 获得节点嵌入
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. 分类器
        x = F.dropout(x, p=0.025, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)

        return x

class GATGraphNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(GATGraphNet, self).__init__()
        self.gat1 = GATConv(11, 16, dropout=0.05)
        self.gat2 = GATConv(16, 64, dropout=0.05)
        self.liner = Linear(64, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.gat1(x, edge_index)#,edge_attr)
        x = F.relu(x)
        x = self.gat2(x, edge_index)#,edge_attr)
        # x = F.relu(x)
        # 2. Readout layer
        #x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.liner(x)
        #x = F.log_softmax(x, dim=1)
        return x
