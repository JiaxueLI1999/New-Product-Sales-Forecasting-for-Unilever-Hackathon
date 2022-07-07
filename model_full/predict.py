import sys
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('Usage: predict.py <product_info> <sales>')
    
    tic = time.perf_counter()
    
    predict_product_info = pd.read_csv(sys.argv[1], index_col = 0)
    predict_sales = pd.read_csv(sys.argv[2], index_col = 0)
    
    predict_product_info.drop(['material_name','material_name_zh',
                               'category', 'bar_code'], axis = 1, inplace = True)
    
    predict_product_info.drop_duplicates(inplace = True)
    predict_sales.drop_duplicates(inplace = True)
    
    # 处理info的重复uuid的问题
    ndf = pd.DataFrame(columns = predict_product_info.columns) #根据df的列名建一个空表ndf
    uuids = set(predict_product_info['uuid'])
    for u in uuids:
        one = predict_product_info.loc[predict_product_info['uuid'] == u] #获取所有uid等于u的行，之后只会保存一行
        #在这里写if然后只保留一行，然后concat到ndf上，实现只保留一行
        olst = list(one['ingredient']) #或者用set
        zero = one.iloc[[0]] #iloc[行号]是series iloc[[行号]]是dataframe
        #zero['name']=str(olst)
        if len(olst) > 1: #等于1的就不用改了
            zero['ingredient'] = str(olst) #or =''.join(olst)
        ndf = pd.concat([ndf,zero]) #把选出来的zero加到ndf里

    
    df_comb = predict_sales.merge(ndf, on='uuid')
    
    final_id = df_comb[['uuid','channel','sales_period_']] #最后有用
    
    ## 成分
    def ingredient(df_in):
        df = df_in.copy()
        s = set()
        for i in tqdm(df.ingredient):
            i = i.replace(" ",'').replace("',",' ').replace("'",'').replace("[",'').replace("]",'')
            i = set(i.split(' '))
            s = set.union(s, i)
        list_igd = list(s) # 共210种成分
        
        count_1 = -1
        for i in tqdm(df.ingredient):
            i = i.replace(" ",'').replace("',",' ').replace("'",'').replace("[",'').replace("]",'')
            i = i.split(' ')
            count_1 += 1
            df['ingredient'][count_1] = i
        
        for colname in tqdm(list_igd):
            count_2 = 0
            df['df_'+colname] = -1
            for sub in df['ingredient']:
                if colname in sub:
                    df['df_'+colname][count_2] = 1
                else:
                    df['df_'+colname][count_2] = 0
                count_2 += 1
        
        df_out = df.drop(['ingredient'],axis=1)
        return df_out

    ## 发行日期
    def launch_date(df_in):
        df = df_in.copy()
        # str to date
        df['launch_date'] = pd.to_datetime(df['launch_date'])
        # find base date
        base_date = df['launch_date'].min() - datetime.timedelta(1)
        # calculate the lenght of launch date
        df['launch_date'] = df['launch_date'] - base_date
        # date to int
        df['launch_date'] = df['launch_date'].map(lambda x: x.days)
    
        df_out = df.copy()
        return df_out

    ## 品牌
    def brand(df_in):
        df = df_in.copy()
        df = pd.concat([df, pd.get_dummies(df['brand'], prefix='brand')], axis=1)
        df_out = df.drop(['brand'],axis=1)
        return df_out

    ## 渠道
    def channel(df_in):
        df = df_in.copy()
        df = pd.concat([df, pd.get_dummies(df['channel'], prefix='channel')], axis=1)
        df_out = df.drop(['channel'],axis=1)
        return df_out
    
    df_proc = df_comb.copy()
    print('Start processing...')
    print('ingredient')
    df_proc = ingredient(df_proc)
    print('launch_date')
    df_proc = launch_date(df_proc)
    print('brand')
    df_proc = brand(df_proc)
    print('channel')
    df_proc = channel(df_proc)
    print('processing finish!')
    
    ## normalization
    df_nor = df_proc.copy()
    Y = df_nor[['uuid','sales_value']]
    X = df_nor.drop(['sales_value','uuid'],axis=1)
    
    ss = StandardScaler()
    std_data = ss.fit_transform(X)
    origin_data = ss.inverse_transform(std_data)
    
    df_std_ = pd.DataFrame(std_data)
    df_std = pd.concat([Y, df_std_], axis=1)
    
    # PCA
    Y = df_std[['uuid','sales_value']]
    X = df_std.drop(['uuid','sales_value'],axis=1)
    
    pca = PCA(n_components=0.9)
    reduced_X = pca.fit_transform(X)
    
    Xp90 = pd.DataFrame(reduced_X)
    con90 = pd.concat([Y, Xp90], axis=1)
    
    # 部署模型
    df_model = con90.copy()

    x = df_model.drop(['uuid','sales_value'],axis=1)
    y = df_model['sales_value']
    xgb = joblib.load('./xgboost.pkl')
    
    predict_result = xgb.predict(x)
    print(xgb)
    # print(predict_result)
    
    #print(final_id)
    final_id.insert(3,'predict_result',predict_result)
    df_final = pd.merge(predict_sales,final_id,on=['uuid','channel','sales_period_'],how='outer')
    df_final.drop('sales_value', axis=1, inplace=True)
    
    df_final['predict_result'].fillna(9.206592402, inplace=True)
    
    df_final.rename({'predict_result':'sales_value'}, inplace=True)
    print(df_final)
    df_final.to_csv('result.csv', index = 0)
    
    toc = time.perf_counter()
    print(f"Inference completed in {toc - tic:0.4f} seconds")
