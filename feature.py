#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pandas
import pickle
import os
from datetime import *

action_path = "./data/user_action_train_50%.txt"
goods_path = "./data/goods_train.txt"

def users_feature(start_date, end_date):
    """得到用户特征：用户ID，用户某段时间购买过的商品数量，用户某段时间点击购买转化率
       并返回"""
    dump_path = './cache/user_%s_%s.pk' %(start_date, end_date)
    if os.path.exists(dump_path):
        user = pickle.load(open(dump_path, 'rb'))
    else:
        action = get_action(start_date, end_date)
        df = pandas.get_dummies(action['action_type'], prefix='action')
        del action['p_id'], action['date']
        action_user = pandas.concat([action['u_id'], df], axis=1)
        action_user = action_user.groupby('u_id', as_index=False).sum()

        # 统计用户购买商品数量
        user = action_user[['u_id','action_1']]
        user.columns = ['u_id','buying_num']
        #print(user.head())

        #计算用户点击—>购买转化率
        user['buying_ratio'] = action_user['action_1'] / (action_user['action_0'] + action_user['action_1'])

        print(user)
        #user.to_csv("./data/user.txt",sep='\t',index=False,encoding='utf-8')
        pickle.dump(user, open(dump_path, 'wb'))

    return user


def get_action(start_date,end_date):
    """得到某段时间的行为数据，并返回"""
    dump_path = './cache/action_%s_%s.pk' %(start_date,end_date)
    if os.path.exists(dump_path):
        action = pickle.load(open(dump_path,'rb'))
    else:
        if os.path.exists('./cache/action.pk'):
            action = pickle.load(open('./cache/action.pk', 'rb'))
        else:
            action = pandas.read_csv(action_path, sep='\t', header=None)
            action.columns = ['u_id', 'p_id', 'action_type', 'date']

            #action = action.sort_values(by=['date','u_id'])
            #action.to_csv('./data/action.txt',sep='\t',index=False,encoding='utf-8')

            pickle.dump(action, open('./cache/action.pk', 'wb'))

        action = action[(action.date >= start_date) & (action.date <= end_date)] # 符合时间范围内数据
        pickle.dump(action, open(dump_path,'wb'))
    #print(action.info())
    #print(action)
    return action


def product_feature(start_date,end_date):
    """得到商品特征：商品ID，商品品牌ID，商品品类ID，商品某段时间销售量，某段时间点击量，商品某段时间点击购买转化率"""
    dump_path = "./cache/product_%s_%s.pk" %(start_date,end_date)
    if os.path.exists(dump_path):
        product = pickle.load(open(dump_path, 'rb'))
    else:
        action = get_action(start_date,end_date)
        df = pandas.get_dummies(action['action_type'], prefix='action')
        del action['u_id'], action['date']

        #统计商品销售量
        product = pandas.concat([action['p_id'], df], axis=1)
        product = product.groupby('p_id', as_index=False).sum()
        product.columns = ['p_id','click_num', 'sales_num']

        # 计算商品点击—>购买转化率
        product['sale_ratio'] = product['sales_num'] / (product['click_num'] + product['sales_num'])

        #商品品牌和类别
        goods_info = pandas.read_csv(goods_path, sep='\t', header=None)
        goods_info.columns = ['p_id', 'band_id', 'cat_id']
        product = pandas.merge(goods_info,product,how='left',on='p_id')
        product = product.fillna(value=0)
        product = product.sort_values(by='p_id')

        print(product)
        #product.to_csv("./data/product.txt",sep='\t',index=False,encoding='utf-8')
        pickle.dump(product, open(dump_path, 'wb'))

    return product


def user_feature(start_date,end_date):
    """统计用户在某段时间内对不同类别不同品牌商品的购买、点击行为。
    统计距预测开始日期1，2，3，5，8，13，21，34天时间内，用户对不同类别、不同品牌商品的点击购买行为累积
     返回两个DF: cat_action, band_action"""
    dump_cat = './cache/user_cat_action_%s_%s.pk'%(start_date,end_date)
    dump_band = './cache/user_band_action_%s_%s.pk'%(start_date,end_date)
    if os.path.exists(dump_cat) and os.path.exists(dump_band):
        cat_action = pickle.load(open(dump_cat,'rb'))
        band_action =pickle.load(open(dump_band,'rb'))
    else:
        action = get_action(start_date,end_date)

        #商品类别，品牌信息
        goods_info = pandas.read_csv(goods_path, sep='\t', header=None)
        goods_info.columns = ['p_id', 'band_id', 'cat_id']

        action = pandas.merge(action,goods_info,how='left',on='p_id')

        del action['p_id']
        #print(action)
        # action：u_id,cat_id, date, action_type

        #统计不同时间段内行为累计
        cat_action = None
        band_action = None
        for i in (0,1,2,4,7,12,20,33):
            #从date_section这天开始计算到end_date，date_section是距预测日期1，2，3，5，...的日期
            date_section = datetime.strptime(end_date,'%m-%d') - timedelta(days=i)
            if date_section < datetime.strptime(start_date,'%m-%d'):
                break #超过了范围

            date_section = date_section.strftime('%m-%d') # change to string
            action_section = action[(action.date >= date_section) & (action.date <= end_date)]  # 符合时间范围内数据
            del action_section['date']
            # print(action_section)

            # 不同类别的行为累积
            df_cat = pandas.get_dummies(action_section['action_type'], prefix='day%s-cat-action' % str(i + 1))
            cat_section = pandas.concat([action_section[['u_id','cat_id']], df_cat], axis=1)
            cat_section = cat_section.groupby(['u_id', 'cat_id'], as_index=False).sum()
            if cat_action is None:
                cat_action = cat_section
            else:
                cat_action = pandas.merge(cat_action, cat_section, how='outer', on=['u_id', 'cat_id'])

            #不同品牌的行为累积
            df_band = pandas.get_dummies(action_section['action_type'], prefix='day%s-band-action' % str(i + 1))
            band_section = pandas.concat([action_section[['u_id','band_id']],df_band], axis=1)
            band_section = band_section.groupby(['u_id','band_id'],as_index=False).sum()
            if band_action is None:
                band_action = band_section
            else:
                band_action = pandas.merge(band_action, band_section, how='outer', on=['u_id', 'band_id'])



        pickle.dump(cat_action,open(dump_cat,'wb'))
        pickle.dump(band_action, open(dump_band, 'wb'))

    print(cat_action.info())
    #print(cat_action)
    print(band_action.info())
    #print(band_action)
    return cat_action, band_action


def action():
    action = pickle.load(open('./cache_full/action.pk','rb'))
    print(action.info())
    print(action[action['action_type'] == 1].count())
