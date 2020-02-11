import numpy as np
import warnings
import cPickle as pickle
warnings.filterwarnings("ignore")


class WFPFeatures(object):
    def __init__(self, trace_data):
        # change list to ndarray
        self.trace_data = np.asarray(trace_data)
        # initiate ts
        self.trace_data[:, 0] -= self.trace_data[0, 0]
        self.in_ = []
        self.out_ = []
        self.in_out()
        self.in_ = np.asarray(self.in_)
        self.out_ = np.asarray(self.out_)
        self.in_interval = []
        self.out_interval = []
        self.total_interval = []
        self.inter_arrival_times()

    def in_out(self):
        """
        dividing trace data into incoming and outgoing data
        :return: incoming array and outgoing array
        """
        self.in_ = self.trace_data[np.where(self.trace_data[:, 1] < 0)[0]]
        self.out_ = self.trace_data[np.where(self.trace_data[:, 1] > 0)[0]]

        # for p in self.trace_data:
        #     if p[1] < 0:
        #         self.in_.append(p)
        #     if p[1] > 0:
        #         self.out_.append(p)

    def inter_pkt_time(self, list_data):
        """
        compute inter-time
        :param list_data: array of [relative_arrive_time, size], size < 0 : coming; size > 0 : outgoing
        :return:
        """
        if len(list_data) < 2:
            return [0]
        times = list_data[:, 0]
        inters = []
        for elem, next_elem in zip(times, times[1:] + [times[0]]):
            inters.append(next_elem - elem)
        if len(inters) == 1:
            return inters
        return inters[:-1]

    def inter_arrival_times(self):
        """
        Compute inter-time lists on incoming, outgoing and total lists
        :return:
        """
        self.in_interval = self.inter_pkt_time(self.in_)
        self.out_interval = self.inter_pkt_time(self.out_)
        self.total_interval = self.inter_pkt_time(self.trace_data)

    def number_pkt_stats(self):
        """
        The total number of packets, along with the number of incoming and outgoing packets
        :return: 3 features
        """
        return len(self.trace_data), len(self.in_), len(self.out_)

    def fraction_of_total(self):
        """
        The number of incoming and outgoing packets as a fraction of the total number of packets
        :return: 2 features
        """
        in_frac = len(self.in_) / float(len(self.trace_data))
        return in_frac, 1 - in_frac

    def avg_pkt_ordering_stats(self):
        """
        Variant of packet ordering features from http://cacr.uwaterloo.ca/techreports/2014/cacr2014-05.pdf
        The total number of packets before it in the sequence
        Burst pattern of in_ in out_ and previous out_
        Burst pattern of out_ in in_ and previous in_
        :return:  4 features
        """
        pre_sum = 0
        out_order = []
        in_order = []
        for p in self.trace_data:
            if p[1] > 0:
                out_order.append(pre_sum)
            if p[1] < 0:
                in_order.append(pre_sum)
            pre_sum += 1
        # void np calculate error, usually do not need
        if len(out_order) == 0:
            out_order.append(0)
        if len(in_order) == 0:
            in_order.append(0)
        return [
            np.mean(in_order),
            np.mean(out_order),
            np.std(out_order),
            np.std(in_order)
        ]

    def chunk(self, seq, num):
        avg = len(seq) / float(num)
        out = []
        for x in xrange(num):
            out.append(seq[int(avg * x):int(avg * (x + 1))])
        return out

    def number_per_sec(self):
        """
        The number of packets per second, along with the mean, standard deviation, min, max, median.
        n <= x < n+1
        :return: 5 features + 20 features
        """
        # timestamp(s)
        times = self.trace_data[:, 0]
        max_time = int(max(times)) + 1
        num_per_sec = [0] * max_time
        for t in times:
            num_per_sec[int(t)] += 1
        alt_per_sec = [sum(x) for x in self.chunk(num_per_sec, 20)]
        stats = [
            np.mean(num_per_sec),
            np.std(num_per_sec),
            np.median(num_per_sec),
            np.min(num_per_sec),
            np.max(num_per_sec)
            # np.sum(alt_per_sec)
        ]
        stats.extend(alt_per_sec)
        return stats

    def first_and_last_30_stats(self):
        """
        Compute the number of incoming and outgoing packets in the first and last 30 packets
        :return: 4 features
        """
        first30 = self.trace_data[:30]
        last30 = self.trace_data[-30:]
        first30in = []
        first30out = []
        for p in first30:
            if p[1] < 0:
                first30in.append(p)
            if p[1] > 0:
                first30out.append(p)
        last30in = []
        last30out = []
        for p in last30:
            if p[1] < 0:
                last30in.append(p)
            if p[1] > 0:
                last30out.append(p)
        return [
            len(first30in),
            len(first30out),
            len(last30in),
            len(last30out)
        ]

    def inter_arrival_stats(self):
        """
        For the total, incoming and outgoing packet streams extract the lists of inter-arrival times between packets
        For each list extract the max, mean, standard deviation, and third quartile
        :return: 12 features
        """
        inter_stats = [
            np.max(self.in_interval),
            np.max(self.out_interval),
            np.max(self.total_interval),
            np.mean(self.in_interval),
            np.mean(self.out_interval),
            np.mean(self.total_interval),
            np.std(self.in_interval),
            np.std(self.out_interval),
            np.std(self.total_interval),
            np.percentile(self.in_interval, 75),
            np.percentile(self.out_interval, 75),
            np.percentile(self.total_interval, 75)
        ]
        return inter_stats

    def time_percentile_stats(self):
        """
        For the total, incoming and outgoing packet sequences
        we extract the first, second, third quartile and total transmission time
        :return: 12 features
        """
        in1 = self.in_[:, 0] if len(self.in_) != 0 else [0]
        out1 = self.out_[:, 0] if len(self.out_) != 0 else [0]
        total1 = self.trace_data[:, 0]
        if len(in1) == 0:
            in1 = np.append(in1, 0)
        if len(out1) == 0:
            out1 = np.append(out1, 0)
        stats = [
            np.percentile(in1, 25),
            np.percentile(in1, 50),
            np.percentile(in1, 75),
            np.percentile(in1, 100),
            np.percentile(out1, 25),
            np.percentile(out1, 50),
            np.percentile(out1, 75),
            np.percentile(out1, 100),
            np.percentile(total1, 25),
            np.percentile(total1, 50),
            np.percentile(total1, 75),
            np.percentile(total1, 100),
        ]
        return stats

    def pkt_concentration_stats(self):
        """
        This subset of features is based on the concentration of outgoing packets feature list.
        The outgoing packets feature list split into 30 evenly sized subsets and sum each subset
        :return: 5 concentration features and 20 alternation features
        """
        subset_size = 30
        chunks = [self.trace_data[x:x + subset_size]
                  for x in xrange(0, len(self.trace_data), subset_size)]
        con = []
        for item in chunks:
            c = 0
            for p in item:
                if p[1] > 0:
                    c += 1
            con.append(c)
        alt_con = [sum(x) for x in self.chunk(con, 20)]
        stats = [
            np.max(con),
            np.min(con),
            np.mean(con),
            np.std(con),
            np.median(con)
            # np.sum(alt_con)
        ]
        stats.extend(alt_con)
        return stats

    def cumul_100(self):
        """
        Deriving n features c1,c2,....cN by sampling the piecewise linear interpolant
        Reference: Website Fingerprinting at Internet Scale
        :return: 100 features
        """
        result = []
        num = 100
        data_len = len(self.trace_data)
        lens = []
        total_length = 0
        sizes = self.trace_data[:, 1]
        # sizes = [x[1] for x in self.trace_data]
        for size in sizes:
            # total_length += -size
            total_length += size
            lens.append(total_length)
        x = []
        xp = np.arange(data_len)
        avg = float(data_len) / num
        for inter in xrange(1, num+1):
            x.append(inter * avg)
        result = np.interp(x, xp, lens)

        # import matplotlib.pyplot as plt
        # plt.plot(xp, lens, '-x')
        # plt.plot(x, result, 'o')
        # plt.show()

        # x_new = np.arange(1, num + 1, 1)
        # if data_len > num:
        #     interval = float(data_len) / num
        #     x_new = np.arange(interval, data_len + 1, interval)
        # y_new = np.interp(x_new, x, lens)
        # result.extend(list(y_new))
        return list(result)

    def packet_ordering(self):
        """
        1. The total number of packets before each outgoing packet. (list size is 300)
        2. The total number of packets between this outgoing packet and previous one. (list size is 300)
        Reference: Effective Attacks and Provable Defenses for Website Fingerprinting
        :return: 600 features
        """
        restult = []
        sizes = self.trace_data[:, 1]
        count = 0
        for i in range(0, len(sizes)):
            if sizes[i] > 0:
                count += 1
                restult.append(i)  #
            if count == 300:
                break
        for i in range(count, 300):
            restult.append(0)

        count = 0
        prevloc = 0
        for i in range(0, len(sizes)):
            if sizes[i] > 0:
                count += 1
                restult.append(i - prevloc)
                prevloc = i
            if count == 300:
                break
        for i in range(count, 300):
            restult.append(0)
        return restult

    # ############## package size features##################
    def pkt_size_stats(self):
        """
        Count the number of packet size in every interval
        inters_out [(0, 64), (64, 128), (128, 256),
                     (256, 512), (512, 1024), (1024, 2048)]
        inters_in [(-2048, -1024), (-1024, -512), (-512, -256),
                    (-256, -128), (-128, -64), (-64, 0)]
        :return: 12 features
        """
        num_inters_out = [0] * 6
        num_inters_in = [0] * 6
        sizes = self.trace_data[:, 1]
        # cumulative or as 1
        for s in sizes:
            if s > 0:
                size_i = int(np.ceil(np.log2(s)) - 6)
                size_i = (0 if size_i < 0 else size_i)
                size_i = (5 if size_i > 5 else size_i)
                num_inters_out[size_i] += 1

            else:
                size_i = int(np.ceil(np.log2(-s)) - 6)
                size_i = (0 if size_i < 0 else size_i)
                size_i = (5 if size_i > 5 else size_i)
                num_inters_in[size_i] += 1

        return num_inters_in + num_inters_out

    def pkt_size_overall_stats(self):
        """
        Compute statistics for sizes of trace data
        i.e. mean, median, std, third-quartile, sum
        :return: 15 features
        """
        sizes = self.trace_data[:, 1]
        # consider add the direction
        # in_sizes = self.in_[:, 1] if len(self.in_) != 0 else [0]
        in_sizes = self.in_[:, 1] * -1 if len(self.in_) != 0 else [0]
        out_sizes = self.out_[:, 1] if len(self.out_) != 0 else [0]
        in_stats = [np.mean(in_sizes), np.median(in_sizes), np.std(in_sizes), np.percentile(in_sizes, 75),
                    np.sum(in_sizes)]
        out_stats = [np.mean(out_sizes), np.median(out_sizes), np.std(out_sizes), np.percentile(out_sizes, 75),
                     np.sum(out_sizes)]
        overall_stats = [np.mean(sizes), np.median(sizes), np.std(sizes), np.percentile(sizes, 75),
                         np.sum(sizes)]
        return in_stats + out_stats + overall_stats

    def unique_pkt_size(self):
        """
        Count the number of per-packet-size
        :return: 2920 features
        """
        uq_size = [0] * 1460 * 2
        sizes = self.trace_data[:, 1]
        for i in xrange(-1460, 1460):
            if i in sizes:
                uq_size.append(1)
            else:
                uq_size.append(0)
        return uq_size

    def get_all_features(self):
        # 3+2+4+(5+20)+4+12+12+(5+20)+100+600+12+15=814
        features = []
        # collect all features
        features.extend(self.number_pkt_stats())
        features.extend(self.fraction_of_total())
        features.extend(self.avg_pkt_ordering_stats())
        features.extend(self.number_per_sec())
        features.extend(self.first_and_last_30_stats())
        features.extend(self.inter_arrival_stats())
        features.extend(self.time_percentile_stats())
        features.extend(self.pkt_concentration_stats())
        features.extend(self.cumul_100())
        features.extend(self.packet_ordering())
        features.extend(self.pkt_size_stats())
        # features.extend(self.unique_pkt_size())
        features.extend(self.pkt_size_overall_stats())
        return features

    def get_cluster_features(self):
        features = []
        features.extend(self.number_pkt_stats())
        features.extend(self.fraction_of_total())
        return features


if __name__ == '__main__':
    # page dataset
    data = pickle.load(open('./google.pickle', 'rb'))
    page_dict_feats = dict()
    # flow dataset
    # data = pickle.load(open('./google_flow.pickle', 'rb'))
    for page in data:
        page_dict_feats[page] = []
        count = 0
        for page_seq in data[page]:
            print page, count
            features = WFPFeatures(page_seq).get_all_features()
            page_dict_feats[page].append(features)
            count += 1
    pickle.dump(page_dict_feats, open('./google_feat.pickle', 'wb'), protocol=2)
