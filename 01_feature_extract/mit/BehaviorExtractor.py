"""
@Created Time : 2022/11/22
@Author  : LiYao
@FileName: BehaviorExtractor.py
@Description:nfstream提取行为特征
@Modified:
    :First modified
    :Modified content:
"""

'''
标签编号：rdp：1，voip：2，youtube：3，netflix：4，rsync：5，scp：6，sftp：7
skype：8，vimeo：9，ssh：10

流媒体 1
chat 2 
c2 3
文件传输 4
voip 5

'''
from nfstream import NFStreamer
import pandas as pd
import os
from NfstreamPlugin import TrafficExtractorPlugin


class BehaviorExtractor(object):
    def __init__(self):
        """"""
        self.pcap_path = './test_pcap/'
        #self.pcap_path = 'F:\\dataset\\111\\'
        self.bpf_filter = 'ip and (tcp or udp) and (!port 53 and !port 5353 and !port 5355) and !host 239.255.255.250'

        pass

    def extract(self):
        """将指定目录pcap中的流量组流，并提取需要的特征至csv文件"""

        print('PCAP PROCESSING...\n')
        pcap_list = []
        for src_pcap in os.listdir(self.pcap_path):
            if os.path.splitext(src_pcap)[1] == '.pcapng' or os.path.splitext(src_pcap)[1] == '.pcap':
                pcap_list.append(src_pcap)

        for index, pcap in enumerate(pcap_list):
            if pcap[9] == 'r' and pcap[10] == 'd' and pcap[11] == 'p':
                self.bpf_filter = 'ip and (tcp port 3389 or udp port 3389)'
            elif pcap[9] == 's' and pcap[10] == 's' and pcap[11] == 'h' or pcap[0] == '4':
                self.bpf_filter = 'ip and (tcp)'
            elif pcap[0] == '5':
                self.bpf_filter = 'ip and (tcp or udp) and (!port 53)'

            my_streamer = NFStreamer(source=self.pcap_path + '/' + pcap,
                                     decode_tunnels=True,
                                     bpf_filter=self.bpf_filter,
                                     promiscuous_mode=True,
                                     snapshot_length=1536,
                                     idle_timeout=1200,
                                     active_timeout=18000,
                                     accounting_mode=0,
                                     udps=TrafficExtractorPlugin(),
                                     n_dissections=0,
                                     statistical_analysis=True,
                                     splt_analysis=0,
                                     n_meters=0,
                                     performance_report=0)

            df = my_streamer.to_pandas()
            if pcap[0] == '1':
                df['label'] = 1
            if pcap[0] == '2':
                df['label'] = 2
            if pcap[0] == '3':
                df['label'] = 3
            if pcap[0] == '4':
                df['label'] = 4
            if pcap[0] == '5':
                df['label'] = 5

            # if pcap[7] == 'r' and pcap[8] == 'd' and pcap[9] == 'p':
            #     df['label'] = 1
            # if pcap[7] == 'v' and pcap[8] == 'o' and pcap[9] == 'i':
            #     df['label'] = 2
            # if pcap[7] == 'y' and pcap[8] == 'o' and pcap[9] == 'u':
            #     df['label'] = 3
            # if pcap[7] == 's' and pcap[8] == 's' and pcap[9] == 'h':
            #     df['label'] = 10
            # if pcap[0] == 't' and pcap[1] == 'm' and pcap[2] == 'a':
            #     df['label'] = 0
            # if pcap[0] == 'w' and pcap[1] == 'i' and pcap[2] == 'k':
            #     df['label'] = 1
            df.to_csv(self.pcap_path + '/' + pcap + '.csv', encoding='utf-8', index=False)

        '''合并生成的多个csv文件'''
        csv_list = []
        data_list = []
        for csv in os.listdir(self.pcap_path):
            if os.path.splitext(csv)[1] == '.csv':
                csv_list.append(self.pcap_path + '/' + csv)

        '''拼接所有的csv文件'''
        for _ in csv_list:
            df = pd.read_csv(_).sample(frac=1.0)
            df_data = df.iloc[:]
            data_list.append(df_data)
        res = pd.concat(data_list)
        # 这里如果已经存在了csv文件，可能会直接追加
        res.to_csv(self.pcap_path + '/merge_data_mit.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    extract_object = BehaviorExtractor()
    extract_object.extract()

