from feature import *
import pandas
from sklearn.model_selection import train_test_split
import xgboost as xgb
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
    train_path = "./cache/train.txt"
    if os.path.exists(train_path):
        feature = pandas.read_csv(train_path,sep=' ')
    else:
        sample1 = make_train_set("01-03","02-05","02-06","02-11")
        sample2 = make_train_set("01-09", "02-11", "02-12", "02-18")
        sample3 = make_train_set("01-16", "02-18", "02-19", "02-25")
        sample4 = make_train_set("01-24", "02-26", "03-01", "03-06")
        sample5 = make_train_set("01-30", "03-06", "03-09", "03-15")
        sample6 = make_train_set("02-07", "03-15", "03-16", "03-22")
        sample7 = make_train_set("02-14", "03-22", "03-23", "03-30")

        feature = pandas.concat([sample1,sample2,sample3,sample4,sample5,sample6,sample7],axis=0)
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


def make_train_label2(tag_start, tag_end):
    action = get_action(tag_start, tag_end)
    product = pandas.read_csv(goods_path, sep='\t', header=None)
    product.columns = ['p_id', 'band_id', 'cat_id']

    action = action[action.action_type == 1]
    action = action.groupby(['u_id','p_id'], as_index=False).sum()
    label = pandas.merge(action, product, how='left', on='p_id')

    #同一个商品
    pro_label = action[['u_id', 'p_id']]
    pro_label['label'] = 1

    #同品牌，同类
    band_cat_label = label.groupby(['u_id','band_id','cat_id'],as_index=False).sum()
    del band_cat_label['p_id']
    band_cat_label['label'] = 0.75

    #同类不同品牌
    cat_label = label.groupby(['u_id','cat_id'], as_index=False).sum()
    del cat_label['p_id']
    del cat_label['band_id']
    cat_label['label'] = 0.5

    #同品牌不同类
    band_label = label.groupby(['u_id', 'band_id'], as_index=False).sum()
    del band_label['p_id']
    del band_label['cat_id']
    band_label['label'] = 0.25

    print(pro_label)
    print(band_cat_label)
    print(cat_label)
    print(band_label)
    #return label
#make_train_label2("03-25","03-31")

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
    start_date = "03-03"
    end_date = "03-24"
    tag_start = "03-25"
    tage_end = "03-31"

    user_index, training_data, label = make_train()
    X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2,
                                                        random_state=0)

    print("Start Training...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'learning_rate': 0.05, 'max_depth': 7, 'alpha':2, 'lambda':3,
             'min_child_weight': 6, 'gamma': 0, 'subsample': 0.9, 'colsample_bytree': 0.9,
             'scale_pos_weight': 1, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 100
    param['eval_metric'] = 'logloss'
    plst = param.items()
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(plst, dtrain, num_round, evallist)
    print("End Training.")

    print("Start Predicting...")
    test_start = "02-23"
    test_end = "03-31"
    user_test, test_data = make_data_set(test_start, test_end)
    test_data = xgb.DMatrix(test_data.values)
    y = bst.predict(test_data)
    user_test['rate'] = y
    print(user_test)

    #user_test['rate'] = numpy.around(user_test['rate'],3)
    del user_test['u_id']
    del user_test['p_id']
    user_test.to_csv('result-proba.txt',sep=' ',index=False,index_label = False,header=False)
    print("End")

#xgboost_pre()





def result_analysis():
    result = pandas.read_csv('result-proba.txt',sep=' ',header=None)
    print(result.describe())
    #result.at[result[0] < 0.1] = 0
    result.at[result[0] >= 0.1] = 1
    print(result[result[0] == 1].count())

    #result = result.apply(lambda x: (x-0.0091)/0.028)
    #result.at[result[0] >= 0.2] = 1
    #print(result)
    #print(result[result[0] == 1].count())
    result.to_csv('result_01.txt',sep=' ',index=False,index_label = False,header=False)

result_analysis()

def train():
    train = pandas.read_csv('./cache_full/train_03-03_03-24.txt',sep=' ')
    print(train.info())
    print(train[train['label'] == 1].count())

#train()