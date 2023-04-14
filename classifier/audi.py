from classifier.base import Classifier

from scipy.fftpack import fft
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

import math

PERIODIC_FLOWS_COUNTER = 0  # f1
PERIODIC_FLOWS_LAYER = 1  # f2
MEAN_PERIOD = 2  # f3
SD_PERIOD = 3  # f4
FLOWS_ONLY_PERIOD = 4  # f5
FLOWS_MULTI_PERIOD = 5  # f6
FLOWS_STATIC_SRC_PORT = 6  # f7
FLOWS_MEAN_PORT_CHANGE = 7  # f8
FLOWS_SD_PORT_CHANGE = 8  # f9
PERIODS_IN_ALL_SUB_CAPTURES = 9  # f10
MEAN_PERIOD_INFER_SUCCESS = 10  # f11
SD_PERIOD_INFER_SUCCESS = 11  # f12
PERIODS_5_29 = 12  # f13
PERIODS_30_59 = 13  # f14
PERIODS_60_119 = 14  # f15
PERIODS_120_600 = 15  # f16
MEAN_R_02_07 = 16  # f17
MEAN_R_07_1 = 17  # f18
MEAN_R_1_2 = 18  # f19
MEAN_R_2 = 19  # f20
SD_R_0_002 = 20  # f21
SD_R_002_01 = 21  # f22
SD_R_01 = 22  # f23
MEAN_RN_02_07 = 23  # f24
MEAN_RN_07_1 = 24  # f25
MEAN_RN_1_2 = 25  # f26
MEAN_RN_2 = 26  # f27
SD_RN_0_002 = 27  # f28
SD_RN_002_01 = 28  # f29
SD_RN_01 = 29  # f30
MEAN_RN_R_0_002 = 30  # f31
MEAN_RN_R_002_01 = 31  # f32
MEAN_RN_R_01 = 32  # f33


class AuDIClassifier(Classifier):
    '''
    Marchal, Samuel, et al. "Audi: Toward autonomous iot device-type 
    identification using periodic communication." IEEE Journal on Selected 
    Areas in Communications 37.6 (2019): 1402-1412.
    '''
    
    def __init__(self, interval=1800):
        super(AuDIClassifier, self).__init__()
        self.tag = 'audi'
        self.interval = interval
        self.selected_features = ['timestamp', 'address_src', 'eth_type', 'address_dst', 'ip_proto', 'tcp_srcport',
                                  'tcp_dstport', 'udp_srcport', 'udp_dstport']

    def get_dataset(self, raw_dataset, generator):
        print("----get_dataset----")
        counter = {address: {} for address in raw_dataset.iot_list.values()}
        print(counter)
        dataset = {address: [] for address in raw_dataset.iot_list.values()}
        # print(dataset)
        instance_index = 0
        # 从csv文件中获得特征
        for t, addr_src, addr_dst, eth_type, ip_proto, tcp_srcport, tcp_dstport, udp_srcport, udp_dstport in generator:
            t = int(float(t))
            # //表示整数除法  对5小时后的数据进行处理
            if instance_index and t // self.interval != instance_index:
                break
                # for address, c in counter.items():
                #     # 判断c是否为空
                #     if c:
                #         # print(address)
                #         feature = self.get_feature(c)
                #         dataset[address].append(feature)
                # instance_index = t // self.interval
                # counter = {address: {} for address in raw_dataset.iot_list.values()}
            if not instance_index:
                instance_index = t // self.interval
            # 第几层的相关信息即对其进行补充，例如2-数据链路层、3-网络层、4-传输层
            if addr_src in counter.keys():
                # print(addr_src)
                # 若在该时刻存在该类型数据，将其置为1，后续同理
                if '2-' + eth_type in counter[addr_src]:
                    counter[addr_src]['2-' + eth_type]['s'][t - instance_index * self.interval] = 1
                else:
                    counter[addr_src]['2-' + eth_type] = {}
                    # 每个流离散化为以每秒一个值采样的二进制时间序列，指示流在 1 秒期间是否包含一个或多个数据包（值 1）或不包含（值 0）
                    counter[addr_src]['2-' + eth_type]['s'] = [0] * self.interval
                    counter[addr_src]['2-' + eth_type]['s'][t - instance_index * self.interval] = 1
                if ip_proto:
                    if '3-' + ip_proto in counter[addr_src]:
                        counter[addr_src]['3-' + ip_proto]['s'][t - instance_index * self.interval] = 1
                    else:
                        counter[addr_src]['3-' + ip_proto] = {}
                        counter[addr_src]['3-' + ip_proto]['s'] = [0] * self.interval
                        counter[addr_src]['3-' + ip_proto]['s'][t - instance_index * self.interval] = 1
                if tcp_dstport:
                    if '4-t-' + tcp_dstport in counter[addr_src]:
                        counter[addr_src]['4-t-' + tcp_dstport]['s'][t - instance_index * self.interval] = 1
                        # 对过程中端口变化情况进行记录，udp同理
                        if tcp_srcport != counter[addr_src]['4-t-' + tcp_dstport]['last_port_src']:
                            counter[addr_src]['4-t-' + tcp_dstport]['last_port_src'] = tcp_srcport
                            counter[addr_src]['4-t-' + tcp_dstport]['port_src_change_interval'].append(
                                t - counter[addr_src]['4-t-' + tcp_dstport]['last_port_src_time'])
                            counter[addr_src]['4-t-' + tcp_dstport]['last_port_src'] = tcp_srcport
                        counter[addr_src]['4-t-' + tcp_dstport]['last_port_src_time'] = t
                    else:
                        counter[addr_src]['4-t-' + tcp_dstport] = {}
                        counter[addr_src]['4-t-' + tcp_dstport]['s'] = [0] * self.interval
                        counter[addr_src]['4-t-' + tcp_dstport]['s'][t - instance_index * self.interval] = 1
                        counter[addr_src]['4-t-' + tcp_dstport]['last_port_src'] = tcp_srcport
                        counter[addr_src]['4-t-' + tcp_dstport]['last_port_src_time'] = t
                        counter[addr_src]['4-t-' + tcp_dstport]['port_src_change_interval'] = []
                if udp_dstport:
                    if '4-u-' + udp_dstport in counter[addr_src]:
                        counter[addr_src]['4-u-' + udp_dstport]['s'][t - instance_index * self.interval] = 1
                        if tcp_srcport != counter[addr_src]['4-u-' + udp_dstport]['last_port_src']:
                            counter[addr_src]['4-u-' + udp_dstport]['last_port_src'] = tcp_srcport
                            counter[addr_src]['4-u-' + udp_dstport]['port_src_change_interval'].append(
                                t - counter[addr_src]['4-u-' + udp_dstport]['last_port_src_time'])
                            counter[addr_src]['4-u-' + udp_dstport]['last_port_src'] = tcp_srcport
                        counter[addr_src]['4-u-' + udp_dstport]['last_port_src_time'] = t
                    else:
                        counter[addr_src]['4-u-' + udp_dstport] = {}
                        counter[addr_src]['4-u-' + udp_dstport]['s'] = [0] * self.interval
                        counter[addr_src]['4-u-' + udp_dstport]['s'][t - instance_index * self.interval] = 1
                        counter[addr_src]['4-u-' + udp_dstport]['last_port_src'] = tcp_srcport
                        counter[addr_src]['4-u-' + udp_dstport]['last_port_src_time'] = t
                        counter[addr_src]['4-u-' + udp_dstport]['port_src_change_interval'] = []
        for address, c in counter.items():
            if c:
                print(address)
                feature = self.get_feature(c)
                dataset[address].append(feature)
        print("----特征转换end----")
        return dataset

    def train_model(self, dataset, training_set_archive):
        x_train, y_train = self.get_training_dataset(dataset, training_set_archive)
        scalar = MinMaxScaler()
        scalar.fit(x_train)
        x_train = scalar.transform(x_train)
        k_nn = KNeighborsClassifier()
        k_nn.fit(x_train, y_train)
        self.model = k_nn
        self.preprocessor = scalar.transform

    @staticmethod
    # 计算周期
    def _f_period(x):
        y = fft(x)  #快速傅里叶变换
        amplitudes = abs(y)   #取绝对值

        t_amplitude = amplitudes.max() * 0.1

        # 可视化操作

        candidate_period = []
        # 确定为局部最大值
        for i in range(1, len(amplitudes) - 1):
            if amplitudes[i] >= t_amplitude and amplitudes[i] > amplitudes[i - 1] and amplitudes[i] > amplitudes[i + 1]:
                candidate_period.append(i)
        candidate_period_t = []
        for i in range(0, len(candidate_period)):
            t = len(x) / candidate_period[i]
            t_upper_bound = int((1.1 * t))
            t_lower_bound = math.ceil(0.9 * t)
            for j in range(t_lower_bound, t_upper_bound):
                candidate_period_t.append(j)
            candidate_period_t = list(set(candidate_period_t))
        return candidate_period_t

    @staticmethod
    def _r_rn(x, i):
        n = len(x)
        if i >= (n - 1) or i < 1:
            return []
        r_yy_i = np.dot(x[i:], x[:n - i])
        r_yy_i_l1 = np.dot(x[i - 1:], x[:n - i + 1])
        r_yy_i_u1 = np.dot(x[i + 1:], x[:n - i - 1])
        if r_yy_i <= r_yy_i_l1 or r_yy_i <= r_yy_i_u1:
            return []
        r = i * r_yy_i / n
        r_n = i * (r_yy_i + r_yy_i_l1 + r_yy_i_u1) / n
        return [r, r_n]

    def flow_data_process(self, x):
        n = len(x)
        result = {}

        # total flow
        total_result = {}
        candidate_period = self._f_period(x)
        # print("候选周期")
        # print(candidate_period)
        for i in candidate_period:
            r_rn_result = self._r_rn(x, i)
            if r_rn_result:
                total_result[i] = r_rn_result
        # top_half_flow
        top_half_flow = x[:n // 2]
        candidate_period = self._f_period(top_half_flow)
        top_half_result = {}
        for i in candidate_period:
            r_rn_result = self._r_rn(top_half_flow, i)
            if r_rn_result:
                top_half_result[i] = r_rn_result

        # middle_half_flow
        middle_half_flow = x[n // 4:n * 3 // 4]
        candidate_period = self._f_period(middle_half_flow)
        middle_half_result = {}
        for i in candidate_period:
            r_rn_result = self._r_rn(middle_half_flow, i)
            if r_rn_result:
                middle_half_result[i] = r_rn_result

        # latter_half_flow
        latter_half_flow = x[n // 2:]
        candidate_period = self._f_period(latter_half_flow)
        latter_half_result = {}
        for i in candidate_period:
            r_rn_result = self._r_rn(latter_half_flow, i)
            if r_rn_result:
                latter_half_result[i] = r_rn_result

        result['total'] = total_result
        result['top_half'] = top_half_result
        result['middle_half'] = middle_half_result
        result['latter_half'] = latter_half_result
        print(result)
        return result

    def get_feature(self, c):
        # print(c)
        feature = [0] * 33
        all_period, all_period_candidate_match = [], []
        all_r_mean, all_r_sd, all_rn_mean, all_rn_sd = [], [], [], []
        src_port_counter = []
        for flow_key, flow in c.items():
            # print(flow)
            # 流数据处理，得到周期以及离散相关rn
            print(flow_key)
            flow_result = self.flow_data_process(flow['s'])
            r_list = []
            rn_list = []
            period_set = set()
            if 'port_src_change_interval' in flow:
                src_port_counter.append(len(flow['port_src_change_interval']))
            for index, v in flow_result['total'].items():
                temporary_counter = 0
                if index in flow_result['top_half']:
                    temporary_counter += 1
                if index in flow_result['middle_half']:
                    temporary_counter += 1
                if index in flow_result['latter_half']:
                    temporary_counter += 1
                if temporary_counter >= 2:
                    # 周期相关参数计算
                    all_period_candidate_match.append(temporary_counter)
                    period_set.add(index)
                    all_period.append(index)
                    r_list.append(v[0])
                    rn_list.append(v[1])
                if temporary_counter == 3:
                    feature[PERIODS_IN_ALL_SUB_CAPTURES] += 1
            if len(period_set)!=0:
                print(period_set)
            # print(all_period)
            if period_set:
                feature[PERIODIC_FLOWS_COUNTER] += 1
                feature[PERIODIC_FLOWS_LAYER] = max(feature[PERIODIC_FLOWS_LAYER], int(flow_key[0]))
                if len(period_set) == 1:
                    feature[FLOWS_ONLY_PERIOD] += 1
                else:
                    feature[FLOWS_MULTI_PERIOD] += 1
            if len(r_list) != 0:
                all_r_mean.append(np.mean(r_list))
                all_rn_mean.append(np.mean(rn_list))
                if len(r_list) > 1:
                    all_r_sd.append(np.std(r_list, ddof=1))
                    all_rn_sd.append(np.std(rn_list, ddof=1))
                else:
                    all_r_sd.append(0)
                    all_rn_sd.append(0)
        print(all_period)
        if all_period:
            feature[MEAN_PERIOD] = np.mean(all_period)
        feature[FLOWS_STATIC_SRC_PORT] = src_port_counter.count(0)
        if src_port_counter:
            feature[FLOWS_MEAN_PORT_CHANGE] = np.mean(src_port_counter)
        if len(src_port_counter) >= 2:
            feature[FLOWS_SD_PORT_CHANGE] = np.std(src_port_counter, ddof=1)
        if len(all_period) >= 2:
            feature[SD_PERIOD] = np.std(all_period, ddof=1)
        if all_period_candidate_match:
            feature[MEAN_PERIOD_INFER_SUCCESS] = np.mean(all_period_candidate_match)
        if len(all_period_candidate_match) >= 2:
            feature[SD_PERIOD_INFER_SUCCESS] = np.std(all_period_candidate_match, ddof=1)
        for i in all_period:
            if i < 5 or i > 600:
                continue
            elif i < 30:
                feature[PERIODS_5_29] += 1
            elif i < 60:
                feature[PERIODS_30_59] += 1
            elif i < 120:
                feature[PERIODS_60_119] += 1
            else:
                feature[PERIODS_120_600] += 1
        for i in all_r_mean:
            if i < 0.2:
                continue
            if i < 0.7:
                feature[MEAN_R_02_07] += 1
            elif i < 1:
                feature[MEAN_R_07_1] += 1
            elif i < 2:
                feature[MEAN_R_1_2] += 1
            else:
                feature[MEAN_R_2] += 1
        for i in all_r_sd:
            if i < 0.02:
                feature[SD_R_0_002] += 1
            elif i < 0.1:
                feature[SD_R_002_01] += 1
            else:
                feature[SD_R_01] += 1
        for i in all_rn_mean:
            if i < 0.2:
                continue
            if i < 0.7:
                feature[MEAN_RN_02_07] += 1
            elif i < 1:
                feature[MEAN_RN_07_1] += 1
            elif i < 2:
                feature[MEAN_RN_1_2] += 1
            else:
                feature[MEAN_RN_2] += 1
        for i in all_rn_sd:
            if i < 0.02:
                feature[SD_RN_0_002] += 1
            elif i < 0.1:
                feature[SD_RN_002_01] += 1
            else:
                feature[SD_RN_01] += 1
        for i in range(0, len(all_r_mean)):
            t = all_rn_mean[i] - all_r_mean[i]
            if t < 0.02:
                feature[MEAN_RN_R_0_002] += 1
            elif t < 0.1:
                feature[MEAN_RN_R_002_01] += 1
            else:
                feature[MEAN_RN_R_01] += 1
        return feature
