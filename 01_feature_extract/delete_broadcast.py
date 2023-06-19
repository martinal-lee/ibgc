"""
@Created Time : 20232/25
@Author  : LiYao
@FileName: delete_broadcast.py
@Description:去除数据集中的广播流
@Modified:
    :First modified
    :Modified content:
"""

import numpy as np
import pandas as pd

def delete_broadcast():
    iscxvpn_data = pd.read_csv('./merge_data_iscx_12labels.csv')
    iscxtor_data = pd.read_csv('./merge_data_tor_14labels.csv')

    # vpn_dstip = iscxvpn_data['dst_ip']
    # vpn_dstmac = iscxvpn_data['dst_mac'].tolist()
    # tor_dstip = iscxtor_data['dst_ip']
    # tor_dstmac = iscxtor_data['dst_mac']
    iscxvpn_data.drop(iscxvpn_data[iscxvpn_data.dst_mac == 'ff:ff:ff:ff:ff:ff'].index.tolist(),inplace=True)
    iscxtor_data.drop(iscxtor_data[iscxtor_data.dst_mac == 'ff:ff:ff:ff:ff:ff'].index.tolist(),inplace=True)

    iscxvpn_data.to_csv('./merge_data_iscx_12labels_delete_broadcast.csv',encoding='utf-8',index=False)
    iscxtor_data.to_csv('./merge_data_tor_14labels_broadcast.csv',encoding='utf-8',index=False)

delete_broadcast()