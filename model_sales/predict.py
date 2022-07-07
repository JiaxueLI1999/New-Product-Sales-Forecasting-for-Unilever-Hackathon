import sys
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import joblib

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('Usage: predict.py <product_info> <sales>')
    
    tic = time.perf_counter()

    predict_product_info = pd.read_csv(sys.argv[1])
    predict_sales = pd.read_csv(sys.argv[2])
    
    predict_sales.drop_duplicates(inplace=True)
    final_id = predict_sales[['uuid','channel','sales_period_']]
    
    ## 渠道
    def channel(df_in):
        df = df_in.copy()
        df = pd.concat([df, pd.get_dummies(df['channel'], prefix='channel')], axis=1)
        df_out = df.drop(['channel'],axis=1)
        return df_out
    
    df_proc = predict_sales.copy()
    print('Start processing...')
    print('channel')
    df_proc = channel(df_proc)
    print('processing finish!')
    
    df_nor = df_proc.copy()

    Y = df_nor[['uuid','sales_value']]
    X = df_nor.drop(['sales_value','uuid'],axis=1)
    
    ss = StandardScaler()
    std_data = ss.fit_transform(X)
    origin_data = ss.inverse_transform(std_data)
    
    df_std_ = pd.DataFrame(std_data)
    df_std = pd.concat([Y, df_std_], axis=1)
    
    df_model = df_std.copy()

    x = df_model.drop(['uuid','sales_value'],axis=1)
    y = df_model['sales_value']
    
    xgb = joblib.load('./xgboost.pkl')
    
    predict_result = xgb.predict(x)
    print(xgb)
    #print(predict_result)
    
    final_id.insert(3,'predict_result',predict_result)
    
    df_final = pd.merge(predict_sales,final_id,on=['uuid','channel','sales_period_'],how='outer')
    df_final.drop('sales_value', axis=1, inplace=True)
    
    df_final['predict_result'].fillna(9.206592402, inplace=True)
    
    df_final.rename({'predict_result':'sales_value'}, axis=1, inplace=True)
    
    df_final.to_csv('result.csv', index = 0)
    
    toc = time.perf_counter()
    print(f"Inference completed in {toc - tic:0.4f} seconds")
