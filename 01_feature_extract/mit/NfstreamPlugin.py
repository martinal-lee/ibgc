"""
@Created Time : 2022/11/22
@Author  : LiYao
@FileName: NfstreamPlugin.py
@Description:nfstream的插件
@Modified:
    :First modified
    :Modified content:
"""

from nfstream import NFPlugin
import binascii


class TrafficExtractorPlugin(NFPlugin):
    """本插件实现获流数据包大小以及方向列表特征"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def decodeLoad(self, data):
        """
        二进制转换成字符串(pip install adafruit-circuitpython-binascii)
        :param data:二进制data
        :return: 字符串data
        """
        str = binascii.b2a_hex(data).decode()
        if str == '00':
            return None
        newLoad = ''
        i = 0
        for j in range(0, len(str), 2):
            newLoad += str[j:j + 2] + ' '
        newLoad = newLoad[:-1]
        return newLoad

    def on_init(self, packet, flow):
        """初始化需要拿到的字段"""
        flow.udps.bi_pkt_size = str()
        flow.udps.bi_flow_pkt_direction = str()
        #flow.udps.bi_pkt_arrive_time = str()
        #flow.udps.bi_flow_syn = str()

    def on_update(self, packet, flow):
        # 包方向
        flow.udps.bi_flow_pkt_direction += str(packet.direction) + ' '
        # 包大小
        flow.udps.bi_pkt_size += str(packet.ip_size) + ' '
        # 每个数据包的到达时间
        #flow.udps.bi_pkt_arrive_time += str(packet.time) + ' '
        # 每个数据包的syn
        #flow.udps.bi_flow_syn = str(packet.syn) + ' '

    def on_expire(self, flow):
        """流过期时的标志"""
        pass
