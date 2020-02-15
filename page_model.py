import cPickle as pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold


def decision_tree_model(feats, labels):
    x_train, x_test, y_train, y_test = train_test_split(feats, labels, test_size=0.25, stratify=labels)
    print y_train.shape, y_test.shape
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    y_predict = dtc.predict(x_test)
    print classification_report(y_test, y_predict)


def kfold_knn_model(feats, labels):
    k = 4
    kf = StratifiedKFold(n_splits=k)
    avg_tp = 0
    avg_fp = 0
    for train_index, test_index in kf.split(feats, labels):
        x_train, y_train = feats[train_index], labels[train_index]
        x_test, y_test = feats[test_index], labels[test_index]
        knc = KNeighborsClassifier(n_neighbors=3)
        knc.fit(x_train, y_train)
        y_predict = knc.predict(x_test)
        # TPR = recall = tp/(tp+fn)
        # FPR = fp/(fp+tn) = 1-negative_recall
        tp = 0
        p_num = 0
        fp = 0
        n_num = 0
        for index in range(y_test.shape[0]):
            if y_test[index] == 0:
                p_num += 1
                if y_predict[index] == 0:
                    tp += 1
            if y_test[index] == 1:
                n_num += 1
                if y_predict[index] == 0:
                    fp += 1
        print classification_report(y_test, y_predict)
        avg_tp += tp * 1.0 / p_num
        avg_fp += fp * 1.0 / n_num
    return avg_tp/k, avg_fp/k


def knn_model(feats, labels):
    x_train, x_test, y_train, y_test = train_test_split(feats, labels, test_size=0.25, stratify=labels)
    # print y_train.shape, y_test.shape
    knc = KNeighborsClassifier(n_neighbors=3)
    knc.fit(x_train, y_train)
    y_predict = knc.predict(x_test)
    # TPR = recall = tp/(tp+fn)
    # FPR = fp/(fp+tn) = 1-negative_recall
    tp = 0
    p_num = 0
    fp = 0
    n_num = 0
    for index in range(y_test.shape[0]):
        if y_test[index] == 0:
            p_num += 1
            if y_predict[index] == 0:
                tp += 1
        if y_test[index] == 1:
            n_num += 1
            if y_predict[index] == 0:
                fp += 1
    print classification_report(y_test, y_predict)
    return tp * 1.0 / p_num, fp * 1.0 / n_num


def svc_model(feats, labels):
    x_train, x_test, y_train, y_test = train_test_split(feats, labels, test_size=0.25, stratify=labels)
    print y_train.shape, y_test.shape
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    lsvc = LinearSVC()
    lsvc.fit(x_train, y_train)
    y_predict = lsvc.predict(x_test)
    print classification_report(y_test, y_predict)


def cal_wfeats(samples):
    feats = list()
    for sample in samples:
        feat = [0 for x in range(len(lcs_kv_list))]
        feat_index = 0
        for (lcs, value) in lcs_kv_list:
            index = 0
            for ts, flow in sample:
                if flow == lcs[index]:
                    index += 1
                    if index == len(lcs):
                        feat[feat_index] = value
                        break
            feat_index += 1
        feats.append(feat)
    return feats


def cal_feats(samples):
    feats = list()
    for sample in samples:
        feat = [0 for x in range(len(lcs_kv_list))]
        feat_index = 0
        for (lcs, value) in lcs_kv_list:
            index = 0
            for ts, flow in sample:
                if index == len(lcs):
                    feat[feat_index] = 1
                    break
                if flow == lcs[index]:
                    index += 1
            feat_index += 1
        feats.append(feat)
    return feats


if __name__ == '__main__':
    data = pickle.load(open('../ray_data/ray_page_dict_flow_del.pickle', 'rb'))
    tpr = 0
    fpr = 0
    c = 0
    for page_name in data.keys():

        samples = pickle.load(open('./' + page_name + '_pos_samples_par.pickle', 'rb'))
        neg_samples = pickle.load(open('./'+page_name+'_neg_samples_par.pickle', 'rb'))
        labels = [0 for x in range(len(samples))]
        labels.extend([1 for x in range(len(neg_samples))])
        samples.extend(neg_samples)
        # samples = np.asarray(samples)
        labels = np.asarray(labels)
        # print samples.shape, labels.shape
        lcs_kv_list = pickle.load(open('./'+page_name+'_lcs_kv_list.pickle', 'rb'))
        # feats = cal_feats(samples)
        feats = cal_wfeats(samples)
        feats = np.asarray(feats)
        print page_name
        # decision_tree_model(feats, labels)
        # TPR, FPR = knn_model(feats, labels)
        TPR, FPR = kfold_knn_model(feats, labels)
        print TPR, FPR
        tpr += TPR
        fpr += FPR
        c += 1
    print tpr, fpr, c
    print tpr/c, fpr/c

