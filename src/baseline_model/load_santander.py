import numpy as np
import pandas as pd
import random
import pickle
import os

random.seed(17)
np.random.seed(17)



def load_train_csv():
    print('Loading train csv')
    train_path = "../../data/Santander/train_ver2.csv"

    target_columns = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
           'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
           'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
           'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
           'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
           'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
           'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
           'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
    other_columns = ['fecha_dato', 'ncodpers'] 
    all_columns = target_columns + other_columns

    dtypes_columns = {}
    for c in target_columns:
        dtypes_columns[c] = np.float16

    #limit_rows = 50
    limit_rows = 50000000

    df = pd.read_csv(train_path, usecols=all_columns, dtype=dtypes_columns, nrows=limit_rows)
    #df_train = pd.read_csv(train_path, usecols=all_columns,  nrows=limit_rows)
    df['fecha_dato'] = pd.to_datetime(df['fecha_dato'])

    #Replace NaN by 0
    df['ind_nomina_ult1'].fillna(0, inplace=True)
    df['ind_nom_pens_ult1'].fillna(0, inplace=True)
    return df
	
	
def load_test_csv():
    print('Loading test csv')
    test_path = "../../data/Santander/test_ver2.csv"
    df_test = pd.read_csv(test_path, usecols=['fecha_dato', 'ncodpers'])
    df_test['fecha_dato'] = pd.to_datetime(df_test['fecha_dato'])
    return df_test