import cPickle as pickle
from joblib import Parallel, delayed
import timeout_decorator
from cal_wfp_features import WFPFeatures
import traceback


def producer(page_seq_dict):
    for page in page_seq_dict:
        for page_index in page_seq_dict[page]:
            yield page, page_index


def seq2features(data):
    page_dict = Parallel(n_jobs=-1)(delayed(extract_features)(page, page_index) for page, page_index in producer(data))
    page_features_dict = dict()
    for page, page_index_features in page_dict:
        if page not in page_features_dict:
            page_features_dict[page] = []
        page_features_dict[page].append(page_index_features)
    return page_features_dict


def extract_features(page, page_index):
    count = 0
    page_index_features = dict()
    for flow in page_index:
        if flow not in page_index_features:
            page_index_features[flow] = []
        for ii, url in enumerate(page_index[flow]):
            print 'Extracting features in %s %s %d' % (page, flow, ii)
            try:
                page_index_features[flow].append(compute_features(url))
            except Exception as e:
                print traceback.print_exc()
                print 'error in extract features of %d  ' % count, e
            finally:
                count += 1
    return page, page_index_features


# @timeout_decorator.timeout(60)
def compute_features(seq):
    return WFPFeatures(seq).get_all_features()


if __name__ == '__main__':
    """
     format is {label:[[ts, len],[ts, len],...], [[], [],...],...}
    """
    page_dict_url = pickle.load(open('F:\\paper\\page_dict_flow_cluster.pickle', 'rb'))
    all_ret = seq2features(page_dict_url)
    pickle.dump(all_ret, open('F:\\paper\\page_dict_flow_feats_cluster.pickle', 'wb'), protocol=2)
    print 'Done'

