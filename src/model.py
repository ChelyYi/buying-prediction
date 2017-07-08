# -*- coding: UTF-8 -*-

# ©copyright Chely_Yi
# github: https://github.com/ChelyYi

from feature import *
import pandas
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import gc

test_item = './data/user_action_test_items.txt'

def make_train_set(start_date,end_date,tag_start, tag_end):
    """训练集，返回特征矩阵和标签"""
    train_path = "./cache/sample_%s_%s.txt" %(start_date,end_date)
    if os.path.exists(train_path):
        sample = pandas.read_csv(train_path,sep=' ')
    else:
        action = get_action(start_date, end_date)
        user_info = users_feature(start_date, end_date)
        product = product_feature(start_date, end_date)
        user_cat_action, user_band_action = user_feature(start_date, end_date)

        feature = action[['u_id', 'p_id']].copy()
        del action
        gc.collect()
        feature = pandas.merge(feature, user_info, how='left', on='u_id')
        del user_info
        gc.collect()
        feature = pandas.merge(feature, product, how='left', on='p_id')
        del product
        gc.collect()
        feature = pandas.merge(feature, user_cat_action, how='left', on=['u_id', 'cat_id'])
        feature = pandas.merge(feature, user_band_action, how='left', on=['u_id', 'band_id'])
        del user_cat_action
        del user_band_action
        gc.collect()

        #处理label
        label = make_train_label(tag_start, tag_end)
        feature = pandas.merge(feature, label, how='left', on=['u_id', 'p_id'])

        #抽样：label1 : label0 = 1: 100（按照整个action的比例）
        feature = feature.fillna(value=0).copy()
        label_1 = feature[feature['label'] == 1].copy()
        label_1 = label_1[ (label_1['day21-cat-action_0'] != 0.0) & (label_1['day21-cat-action_1'] != 0.0) ]

        label0 = feature[feature['label'] == 0].copy()
        del feature
        gc.collect()

        label1_num = label_1['label'].count()
        label0_num = label1_num * 100
        print("Label 1: " + str(label1_num) +" Label 0: "+ str(label0_num))

        print("Sampling")
        label_0 = label0.sample(label0_num)

        sample = pandas.concat([label_1,label_0])
        sample.to_csv(train_path,sep=' ',index=False,encoding='utf-8')

    print("Train Set:")
    print(sample.info())
    print(sample)

    return sample


def make_train():
    print("Make train set...")
    train_path = "./cache/train1.txt"
    if os.path.exists(train_path):
        feature = pandas.read_csv(train_path,sep=' ')
    else:
        sample1 = make_train_set("01-03","01-23","01-24","01-26")
        sample2 = make_train_set("01-08", "01-28", "01-29", "01-31")
        sample3 = make_train_set("01-11", "01-31", "02-01", "02-03")
        sample4 = make_train_set("01-14", "02-03", "02-04", "02-06")
        sample5 = make_train_set("01-17", "02-06", "02-07", "02-09")
        sample6 = make_train_set("01-20", "02-09", "02-10", "02-12")
        sample7 = make_train_set("01-23", "02-12", "02-13", "02-15")
        sample8 = make_train_set("01-27", "02-16", "02-17", "02-19")
        sample9 = make_train_set("02-03", "02-23", "02-24", "02-26")
        sample10 = make_train_set("03-09", "03-29", "03-30", "03-31")

        feature = pandas.concat([sample1,sample2,sample3,sample4,sample5,sample6,sample7,sample8,sample9,sample10],axis=0)
        feature.to_csv(train_path,sep=' ',index = False,encoding='utf-8')


    label = feature['label'].copy()
    ID = feature[['u_id', 'p_id']].copy()
    del feature['label']
    del feature['u_id']
    del feature['p_id']
    return ID,feature,label


def make_train_label(tag_start, tag_end):
    action = get_action(tag_start, tag_end)
    action = action[action.action_type == 1]
    action = action.groupby(['u_id','p_id'], as_index=False).sum()
    action['label'] = 1
    label = action[['u_id', 'p_id', 'label']]

    print(label)
    return label


def make_data_set(start_date,end_date):
    """数据集，待预测内容"""
    print("Make test set...")
    test_path = "./cache/test_%s_%s.txt" % (start_date, end_date)
    if os.path.exists(test_path):
        test = pandas.read_csv(test_path,sep=' ')
    else:
        test_ID = pandas.read_csv(test_item,sep='\t',header=None)
        test_ID.columns = ['u_id','p_id','na']
        del test_ID['na']

        users = users_feature(start_date, end_date)
        product = product_feature(start_date, end_date)
        user_cat_action, user_band_action = user_feature(start_date, end_date)

        test = pandas.merge(test_ID,users,how='left',on='u_id')
        test = pandas.merge(test, product, how='left', on='p_id')
        test = pandas.merge(test, user_cat_action, how='left', on=['u_id','cat_id'])
        test = pandas.merge(test, user_band_action, how='left', on=['u_id', 'band_id'])
        test = test.fillna(value=0)
        test.to_csv(test_path,sep=' ',index=False,encoding='utf-8')

        del users
        del product
        del user_cat_action
        del user_band_action
        gc.collect()

    print("Test set:")
    print(test)
    test_ID = test[['u_id','p_id']]
    del test['u_id']
    del test['p_id']
    return test_ID, test


def xgboost_pre():

    user_index, training_data, label = make_train()
    X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2,
                                                        random_state=107)

    print("Start Training...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'learning_rate': 0.05, 'max_depth': 5, 'alpha':0, 'lambda':1,
             'min_child_weight': 5, 'gamma': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'scale_pos_weight': 1, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 2000
    param['eval_metric'] = 'auc'
    plst = param.items()
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(plst, dtrain, num_round, evallist)
    print("End Training.")

    print("Start Predicting...")
    test_start = "03-11"
    test_end = "03-31"
    user_test, test_data = make_data_set(test_start, test_end)
    test_data = xgb.DMatrix(test_data.values)
    y = bst.predict(test_data)
    user_test['rate'] = y
    print(user_test)

    #user_test['rate'] = numpy.around(user_test['rate'],3)
    del user_test['u_id']
    del user_test['p_id']
    user_test.to_csv('result_.txt',sep=' ',index=False,index_label = False,header=False)
    print("End")

#xgboost_pre()
#make_train()

def modify_train():
    path = './cache/train1.txt'
    train = pandas.read_csv(path,sep=' ')
    train_1 = train[train['label'] == 1]
    train_0 = train[train['label'] == 0]
    train_1 = train_1[(train_1['day21-cat-action_0'] != 0.0) | (train_1['day21-cat-action_1'] != 0.0)]
    print(train_1.count(),)
    train = pandas.concat([train_0,train_1],axis=0)
    train.to_csv(path,sep=' ',index=False,encoding='utf-8')


def result_analysis(path):
    result = pandas.read_csv(path,sep=' ',header=None)
    print(result.describe())
    #result.at[result[0] < 0.1] = 0
    print(result[result[0] >= 0.1].count())
    print(result[result[0] >= 0.2].count())
    print(result[result[0] >= 0.3].count())
    print(result[result[0] >= 0.4].count())
    print(result[result[0] >= 0.5].count())
    print(result[result[0] >= 0.6].count())
    print(result[result[0] >= 0.7].count())
    print(result[result[0] >= 0.8].count())
    print(result[result[0] >= 0.9].count())

    #result = result.apply(lambda x: (x-0.0091)/0.028)
    result.at[result[0] >= 0.2] = 1
    #print(result)
    #print(result[result[0] == 1].count())
    result.to_csv('result_2000-01.txt',sep=' ',index=False,index_label = False,header=False)
