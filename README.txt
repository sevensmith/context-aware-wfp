context-aware-wfp
ray_page_dict_flow_feats_del.pickle    # traffic consists of 20 monitered websites caputured through v2ray
cal_wfp_features.py    # calculate the features of [ts, directions+size] sequence
page_dict_flow2features.py    # call the WFPFeatures class to calculate the features of flows in every website traffic
rf_knn_cal_proba.py    # generate the flow classifier and the W2I of flows
cal_lcs.py   # calculate the longest common sequences of per access traffic that consisits of flows of the website
page_model.py    # use weight-knn to train and test website classifier
