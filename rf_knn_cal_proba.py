# coding=utf-8

import cPickle as pickle
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support


def knn_model(train_leaves, train_l, test_page_flow):
    global TRAIN_INSTANCES_NUM
    # 将k设置为训练样本数量-1
    knc = KNeighborsClassifier(n_neighbors=TRAIN_INSTANCES_NUM-1)
    t1 = time.time()
    knc.fit(train_leaves, train_l)
    print('knn train time', time.time()-t1)
    # pickle.dump(knc, open('/data/smw/knn.pickle', 'wb'), protocol=2)
    # joblib.dump(knc, './knn_proba'+suffix+'.m')
    # 使用训练样本计算概率
    proba = knc.predict_proba(train_leaves)
    page_flow_proba = []
    tmp_page_flow = []
    index = 0
    for proba_ in proba:
        if len(tmp_page_flow) == TRAIN_INSTANCES_NUM:
            page_flow_proba.append(np.percentile(tmp_page_flow, 50))
            index += 1
            tmp_page_flow = []
        if index >= len(test_page_flow):
            break
        tmp_page_flow.append(proba_[index])
    if len(tmp_page_flow) == TRAIN_INSTANCES_NUM:
        page_flow_proba.append(np.percentile(tmp_page_flow, 50))

    np.savetxt('./flow_proba'+suffix+'.txt', page_flow_proba, fmt='%s')
    # pickle.dump(page_flow_proba, open('./flow_proba'+suffix+'.pickle', 'wb'), protocol=2)

    # page0 = test_page_flow[0][0]
    # tmp_proba = 0.0
    # page_proba = dict()
    # page_proba_txt = []
    # for index in xrange(len(test_page_flow)):
    #     page = test_page_flow[index][0]
    #     if page != page0:
    #         page_proba[page0] = tmp_proba
    #         page_proba_txt.append((page0, tmp_proba))
    #         tmp_proba = 0.0
    #         page0 = page
    #     tmp_proba += page_flow_proba[index]
    # page_proba_txt.append((page0, tmp_proba))
    # page_proba[page0] = tmp_proba
    # np.savetxt('./page_proba'+suffix+'.txt', page_proba_txt, fmt='%s')
    # pickle.dump(page_proba, open('./page_proba'+suffix+'.pickle', 'wb'), protocol=2)

    return page_flow_proba


def rf_model(flows, labels, features, test_l, test_f):
    global CLASS_NUM, TEST_INSTANCES_NUM, TEST_NEG_INSTANCES_NUM, TRAIN_INSTANCES_NUM, TRAIN_NEG_INSTANCES_NUM
    """
    Training and testing based on Random-Forest
    :param labels:
    :param features:
    :param test_l:
    :param test_f:
    :return:
    """
    tmp_page = dict()
    tmp_i = 0
    page_list = []
    for flow in flows:
        if flow[0] not in tmp_page:
            tmp_page[flow[0]] = tmp_i
            tmp_i += 1
            page_list.append(flow[0])
    np.savetxt('./page_list'+suffix+'.txt', page_list, fmt='%s')
    flow_map_page = dict()
    for index in xrange(len(test_l)):
        if OPEN_WORLD and index == len(flows) * TEST_INSTANCES_NUM:
            flow_map_page[str(test_l[index])] = tmp_i
            break
        flow_map_page[str(test_l[index])] = tmp_page[flows[index/TEST_INSTANCES_NUM][0]]
    print 'Negative page index: ', tmp_i
    print "Train ", features.shape, "Test", test_f.shape
    t1 = time.time()
    rf = RandomForestClassifier(
            n_estimators=500, criterion="entropy", n_jobs=-1, oob_score=True, max_features='log2')
    rf.fit(features, labels)
    print "OOB_Score is ", rf.oob_score_, "Build time", time.time() - t1
    print "Build time", time.time() - t1
    importances = rf.feature_importances_
    index_fi = zip(range(features.shape[1]), importances)
    index_fi.sort(key=lambda x: x[1], reverse=True)
    np.savetxt('./index_fi'+suffix+'.txt', index_fi, fmt='%s')
    joblib.dump(rf, './rf'+suffix+'.m')

    train_leaves = rf.apply(features)
    knn_model(train_leaves, labels, flows)

    t2 = time.time()
    pred_l = rf.predict(test_f)
    print "Predict time", time.time() - t2

    p, r, f1, s = precision_recall_fscore_support(test_l, pred_l)
    print 'average TPR: ', np.average(r[:-1], weights=s[:-1])
    print 'fpr: ', 1-r[-1]
    print classification_report(test_l, pred_l)
    # cnf_matrix = confusion_matrix(test_l, pred_l)

    # page0 = flows[0][0]
    # record = []
    # for i in xrange(1, len(flows)):
    #     if flows[i][0] != page0:
    #         record.append(i)
    #         page0 = flows[i][0]
    # record.append(len(flows))
    #
    # pred_label = []
    # test_label = []
    # index = 0
    # for i in record:
    #     # one = test_l[index * TEST_INSTANCES_NUM]
    #
    #     for k in xrange(TEST_INSTANCES_NUM):
    #         test_label.append(flows[index][0])
    #
    #         pred_dict = dict()
    #         pred_set = set()
    #
    #         for j in xrange(index, i):
    #             pred = pred_l[k + j * TEST_INSTANCES_NUM]
    #             if pred == len(flows):
    #                 continue
    #             page = flows[pred][0]
    #             flow = flows[pred][1]
    #             if (page, flow) not in pred_set:
    #                 if page not in pred_dict:
    #                     pred_dict[page] = 0.0
    #                 pred_dict[page] += page_flow_proba[pred]
    #                 pred_set.add((page, flow))
    #         if len(pred_dict) == 0:
    #             pred_label.append('negative')
    #         else:
    #             for k in pred_dict.keys():
    #                 pred_dict[k] = pred_dict[k] / page_proba_dict[k]
    #             pred_list = sorted(pred_dict.iteritems(), key=lambda x: x[1], reverse=True)
    #             # print test_label, pred_list
    #             if pred_list[0][1] >= 0.3:
    #                 pred_label.append(pred_list[0][0])
    #             else:
    #                 pred_label.append('negative')
    #
    #     if i == len(flows):
    #         break
    #     index = i
    #
    # print classification_report(test_label, pred_label)

    # for i in xrange(len(test_l)):
    #     test_l[i] = flow_map_page[str(test_l[i])]
    #     pred_l[i] = flow_map_page[str(pred_l[i])]
    # np.savetxt('./ow30_test_flows.txt', flows, fmt='%s')
    # np.savetxt('./t_ow30_test_l_del.txt', test_l, fmt='%s')
    # np.savetxt('./t_ow30_pred_l_del.txt', pred_l, fmt='%s')
    # print cal_precision_recall(tmp_i, flows, test_l, pred_l)


def cal_precision_recall(neg_i, test_page_flow, test_l, pred_l):
    global CLASS_NUM, TEST_INSTANCES_NUM, TEST_NEG_INSTANCES_NUM, TRAIN_INSTANCES_NUM, TRAIN_NEG_INSTANCES_NUM
    precision_dict = dict()
    average_precision = 0.0
    for i in xrange(len(pred_l)):
        if OPEN_WORLD and pred_l[i] == neg_i:
            continue
        if pred_l[i] not in precision_dict:
            precision_dict[pred_l[i]] = [0, 0]
        precision_dict[pred_l[i]][0] += 1
        if pred_l[i] == test_l[i]:
            precision_dict[pred_l[i]][1] += 1
    for key in precision_dict.keys():
        precision_dict[key] = precision_dict[key][1] * 1.0 / precision_dict[key][0]
        average_precision += precision_dict[key]
    print precision_dict
    print average_precision / len(precision_dict)
    page0 = test_page_flow[0][0]
    record = []
    for i in xrange(1, len(test_page_flow)):
        if test_page_flow[i][0] != page0:
            record.append(i)
            page0 = test_page_flow[i][0]
    record.append(len(test_page_flow))
    page_predict = dict()
    average_recall = 0.0
    index = 0
    for i in record:
        one = test_l[index * TEST_INSTANCES_NUM]
        page_predict[one] = 0
        for k in xrange(TEST_INSTANCES_NUM):
            for j in xrange(index, i):
                if pred_l[k + j * TEST_INSTANCES_NUM] == one:
                    page_predict[one] += 1
                    break
        if i == len(test_page_flow):
            break
        index = i
    for page in page_predict:
        page_predict[page] = float(page_predict[page]) / TEST_INSTANCES_NUM
        average_recall += page_predict[page]
    print page_predict
    print average_recall / len(page_predict)
    return average_precision / len(precision_dict), average_recall / len(page_predict)


def truncate_data():
    """
    The number of samples for each class is not equal.
    For fairness, we select the same number instances for every training class randomly.
    Every time we run this function, samples will be shuffled.
    :return:
    """

    global CLASS_NUM, TEST_INSTANCES_NUM, TEST_NEG_INSTANCES_NUM, TRAIN_INSTANCES_NUM, TRAIN_NEG_INSTANCES_NUM, OPEN_WORLD
    labels, features = [], []
    test_labs, test_feats = [], []
    index_l = 0
    page_flow = []

    # shuffle dict keys
    keys = data.keys()
    for page in keys:
        flows = data[page][0].keys()
        for flow in flows:
            for i in xrange(len(data[page])):
                if i < TRAIN_INSTANCES_NUM:
                    features.append(data[page][i][flow][0])
                    labels.append(index_l)
                else:
                    test_feats.append(data[page][i][flow][0])
                    test_labs.append(index_l)
            page_flow.append((page, flow))
            index_l += 1

    np.savetxt('./page_flow'+suffix+'.txt', page_flow, fmt='%s')
    pickle.dump(page_flow, open('./page_flow'+suffix+'.pickle', 'wb'), protocol=2)
    print 'positive samples numbers: train %d, test %d' % (len(labels), len(test_labs))

    if OPEN_WORLD:
        negative_samples = neg_data[:]
        np.random.shuffle(negative_samples)
        neg_train_samples = negative_samples[:TRAIN_NEG_INSTANCES_NUM]
        neg_test_samples = negative_samples[TRAIN_NEG_INSTANCES_NUM:TRAIN_NEG_INSTANCES_NUM + TEST_NEG_INSTANCES_NUM]
        features.extend(neg_train_samples)
        labels.extend([index_l] * len(neg_train_samples))
        test_feats.extend(neg_test_samples)
        test_labs.extend([index_l] * len(neg_test_samples))
        print 'negative samples numbers: train %d, test %d' % (len(neg_train_samples), len(neg_test_samples))
    # Training samples
    labels = np.asarray(labels)
    features = np.asarray(features)
    features = np.nan_to_num(features)

    # Testing samples
    test_feats = np.asarray(test_feats)
    test_labs = np.asarray(test_labs)
    test_feats = np.nan_to_num(test_feats)

    if len(test_labs) == 0:
        return page_flow, labels, features[:, sp1:sp2], test_labs, test_feats
    print labels.shape, features.shape, test_labs.shape, test_feats.shape
    return page_flow, labels, features[:, sp1:sp2], test_labs, test_feats[:, sp1:sp2]


if __name__ == '__main__':
    CLASS_NUM = 30
    TRAIN_INSTANCES_NUM = 70
    TRAIN_NEG_INSTANCES_NUM = 6000
    TEST_INSTANCES_NUM = 20
    TEST_NEG_INSTANCES_NUM = 10000
    OPEN_WORLD = True
    suffix = '_del'
    # suffix = '_cluster'

    sp1 = 0  # start index of feature
    sp2 = 814  # end index of feature

    data = pickle.load(open('F:\\paper\\v2ray\\ray_page_dict_flow_feats'+suffix+'.pickle', 'rb'))
    neg_data = pickle.load(open('F:\\paper\\v2ray\\ray_neg_flow_feats'+suffix+'.pickle', 'rb'))

    flows, train_labs, train_feats, test_labs, test_feats = truncate_data()
    rf_model(flows, train_labs, train_feats, test_labs, test_feats)
