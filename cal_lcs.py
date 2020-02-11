# coding=utf8
import cPickle as pickle
import numpy as np


# 目前只考虑了flow的name，因为ts,len等特征在识别flow的时候已经考量过，如果需要更多flow的特征，就是用Flow类去判等
class Flow(object):
    def __init__(self, name='', ts=0.0, length=0):
        self.name = name
        self.ts = ts
        self.length = length

    def equals(self, f):
        return self.name == f.name and abs(self.ts-f.ts) <= 0.1 and abs(self.length-f.length) <= 10


# 只考虑每个page中只出现一次域名的flow
def cal(page_name, page, page_flow_dict_proba):
    visit_count = len(page)
    flow_count = len(page[0])
    # n次访问page组成的flow序列
    res = list()
    # i为访问web次数索引
    for i in xrange(visit_count):
        # 对每一次访问得到的flow按照flow开始时间进行排序 tmp是tuple类型的，(域名，[[timestamp, len],...])
        tmp = sorted(page[i].iteritems(), key=lambda x: x[1][0][0][0])
        flow_list = []
        for ttmp in tmp:
            # f = Flow(ttmp[0], ttmp[1][0][0][0], ttmp[1][0][0][1])
            flow_list.append(ttmp[0])
        res.append(flow_list)
    # 对n次访问page组成的flow序列，排列组合进行lcs的提取
    lcs_all = dict()

    for i in xrange(visit_count):
        for j in xrange(i+1, visit_count):
            lcs_tmp = [[0 for s in range(flow_count+1)] for s in range(flow_count+1)]
            lcs_index = [[0 for s in range(flow_count)] for s in range(flow_count)]
            for k in xrange(flow_count):
                for l in xrange(flow_count):
                    # 判断两个flow相等的条件
                    if res[i][k] == res[j][l]:
                        lcs_tmp[k+1][l+1] = lcs_tmp[k][l]+1
                        lcs_index[k][l] = 0
                    elif lcs_tmp[k][l+1] > lcs_tmp[k+1][l]:
                        lcs_tmp[k+1][l+1] = lcs_tmp[k][l+1]
                        lcs_index[k][l] = 1
                    elif lcs_tmp[k][l+1] < lcs_tmp[k+1][l]:
                        lcs_tmp[k+1][l+1] = lcs_tmp[k+1][l]
                        lcs_index[k][l] = 2
                    else:
                        lcs_tmp[k+1][l+1] = lcs_tmp[k+1][l]
                        lcs_index[k][l] = 3
            lcs_list = list()
            lcs_print(flow_count-1, flow_count-1, lcs_index, res[i], lcs_list, list())
            for lcs in lcs_list:
                if lcs in lcs_all:
                    lcs_all[lcs] += 1
                else:
                    lcs_all[lcs] = 1

    sort_lcs_kv = top_k(page_name, lcs_all, page_flow_dict_proba, 10)
    # 检查10个lcs中 是否包括所有flow
    # 并没有包括所有的flow 是否应该把每个flow作为lcs加入sp中  或者 加入不存在的flow
    lcs_record = set()
    for lcs in sort_lcs_kv:
        lcs_record.add(lcs[0])
        # for flow in lcs[0]:
        #     if len(lcs_record) == 16:
        #         break
        #     if flow in page[0]:
        #         lcs_record.add(flow)
    # 将单独的flow也加入到lcs集合
    for flow in page[0].keys():
        extra_lcs = (flow,)
        if extra_lcs not in lcs_record:
            sort_lcs_kv.append((extra_lcs, cal_v(visit_count*(visit_count-1)/2, page_flow_dict_proba[(page_name, flow)])))
    sort_lcs_kv.sort(key=lambda x: x[1], reverse=True)
    return sort_lcs_kv
    # pickle.dump(sort_lcs_kv, open('./'+page_name+'_lcs_kv_list.pickle', 'wb'), protocol=2)
    # res = [x[0] for x in res]
    # return res


# 因为lcs不止一个，通过回溯长度相同但组成不同的lcs
def lcs_print(i, j, lcs_index, flow, lcs_list, lcs):
    if i < 0 or j < 0:
        lcs_list.append(tuple(lcs))
        return
    if lcs_index[i][j] == 0:
        lcs.insert(0, flow[i])
        lcs_print(i-1, j-1, lcs_index, flow, lcs_list, lcs)
        lcs.remove(flow[i])
    elif lcs_index[i][j] == 1:
        lcs_print(i-1, j, lcs_index, flow, lcs_list, lcs)
    elif lcs_index[i][j] == 2:
        lcs_print(i, j-1, lcs_index, flow, lcs_list, lcs)
    else:
        lcs_print(i-1, j, lcs_index, flow, lcs_list, lcs)
        lcs_print(i, j-1, lcs_index, flow, lcs_list, lcs)


# 计算lcs的value，进行排序，得到top k的lcs
def top_k(page_name, lcs_dict_count, page_flow_dict_proba, k):
    lcs_kv = dict()
    for lcs in lcs_dict_count:
        sum = 0
        for cur_flow in lcs:
            cur_proba = page_flow_dict_proba[(page_name, cur_flow)]
            sum += cur_proba
        lcs_kv[lcs] = cal_v(lcs_dict_count[lcs], sum)
    sort_lcs_kv = sorted(lcs_kv.iteritems(), key=lambda x: x[1], reverse=True)
    return sort_lcs_kv[:k]


def cal_v(count, weight):
    return count**0.5 * weight


def read_flow_proba(page_flow, proba):
    page_flow_dict_proba = dict()
    # print page_flow.shape[0]
    for index in xrange(page_flow.shape[0]):
        page_flow_dict_proba[(page_flow[index][0], page_flow[index][1])] = float(proba[index])
    return page_flow_dict_proba


if __name__ == '__main__':
    suffix = '_del'
    data = pickle.load(open('F:\\paper\\page_dict_flow'+suffix+'.pickle', 'rb'))
    page_flow = np.loadtxt('./page_flow'+suffix+'.txt', dtype=str)
    for k in xrange(5, 71, 5):
        proba = np.loadtxt('./flow_proba'+suffix+'_'+str(k)+'.txt', dtype=str)
        page_flow_dict_proba = read_flow_proba(page_flow, proba)
        for key in data.keys():
            # if key == 'www.txxx.com':
            sort_lcs_kv = cal(key, data[key], page_flow_dict_proba)
            pickle.dump(sort_lcs_kv, open('./' + key + '_lcs_kv_list_'+str(k)+'.pickle', 'wb'), protocol=2)
    print 'done'
