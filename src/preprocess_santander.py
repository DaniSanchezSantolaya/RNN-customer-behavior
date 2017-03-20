import numpy as np
import pandas as pd
import random
import pickle
import os
from scipy import sparse

random.seed(17)
np.random.seed(17)


target_columns = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
           'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
           'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
           'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
           'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
           'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
           'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
           'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

#aux_features: segmento, renta, indext, tiprel_1mes, age
def load_train_csv():
    print('Loading train csv')
    train_path = "../data/Santander/train_ver2.csv"

    
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
    
    #Compute month timestamp
    initial_date = df['fecha_dato'].min()
    df['num_month'] = df['fecha_dato'] - initial_date
    df['num_month'] = df['num_month']/np.timedelta64(1, 'M')
    df['num_month'] = df['num_month'].round()
    df['num_month'] = df['num_month'].astype(np.int16)
    
    #Create interactoins
    df = mark_interactions(df)
    
    #Compute difference in months between interactions
    times_last_int = {}
    i = 0
    grouped = df.groupby('ncodpers')
    for ncodpers,group in grouped:
        time_from_last_interaction = np.zeros(len(group))
        time_from_last_interaction[group['b_interaction'] == 1] = group[group['b_interaction'] == 1]['num_month'].astype(np.float16).diff()
        #group.loc[group['b_interaction'] == 1,'time_from_last_interaction'] = time_from_last_interaction
        times_last_int[ncodpers] = time_from_last_interaction
        i = i + 1
        if i % 100000 == 0:
            print(i)
            
    values_time_last_int = np.zeros(len(df), dtype=np.float16)
    current_index = {}
    i = 0
    for ncodpers in df.ncodpers:
        if not ncodpers in current_index:
            current_index[ncodpers] = 0
        c_idx = current_index[ncodpers]
        current_index[ncodpers] = current_index[ncodpers] + 1
        values_time_last_int[i] = times_last_int[ncodpers][c_idx]
        if(i % 2000000 == 0):
            print(i)
        i = i + 1
    df['time_from_last_interaction'] = values_time_last_int.astype(np.float16)
    df['time_from_last_interaction'].fillna(0, inplace=True)
    df['time_from_last_interaction'] = df['time_from_last_interaction'].astype(np.int8)
    
    #Save pickle of preprocessed data frame
    save_train_pickle_interactions(df)
    
    
    return df
	
	
def load_test_csv():
    print('Loading test csv')
    test_path = "../data/Santander/test_ver2.csv"
    df_test = pd.read_csv(test_path, usecols=['fecha_dato', 'ncodpers'])
    df_test['fecha_dato'] = pd.to_datetime(df_test['fecha_dato'])
    return df_test
    
def build_train_and_test(df, df_test, representation, max_interactions, aux_features, padding):

    if representation == 1:
        X_train, Y_train, X_test, X_local_test, Y_local_test = build_rep_1(df, df_test, max_interactions, aux_features)
    if representation == 2: #Same than 1, but predicting adds instead of states of the products
        X_train, Y_train, X_test, X_local_test, Y_local_test = build_rep_2(df, df_test, max_interactions, aux_features)
    elif representation == 4:
        print('Build rep 4')
        #df = mark_interactions(df)
        X_train, Y_train, X_test, X_local_test, Y_local_test = build_rep_4(df, df_test, max_interactions, aux_features, padding)
    return X_train, Y_train, X_test, X_local_test, Y_local_test
    
def mark_interactions(df):

    if not 'b_interaction' in df:
    
        df['b_interaction'] = np.zeros(len(df), dtype=np.int8)
        columns_pos_interaction = []
        columns_neg_interaction = []
        for prod in target_columns:
            df[prod + '_pos_interaction'] = np.zeros(len(df), dtype=np.int8)
            df[prod + '_neg_interaction'] = np.zeros(len(df), dtype=np.int8)
            columns_pos_interaction.append(prod + '_pos_interaction')
            columns_neg_interaction.append(prod + '_neg_interaction')
        
        print('Marking interactions')
        grouped = df.groupby('ncodpers')
        i = 0
        b_interactions = {}
        pos_interactions = {}
        neg_interactions = {}
        for ncodpers,group in grouped:
            interactions = group[target_columns].diff()
            pos_interactions[ncodpers] = (interactions == 1).values.astype(np.int8)
            neg_interactions[ncodpers] = (interactions == -1).values.astype(np.int8)
            b_interactions[ncodpers] = np.any(np.logical_or(interactions == 1, interactions == -1)  , axis=1).astype(np.int8)
   
            i = i + 1

            if(i % 200000 == 0):
                print(i)
                
        values_b_interactions = np.zeros(len(df), dtype=np.int8)
        values_pos_interactions= np.zeros((len(df), len(columns_pos_interaction)), dtype=np.int8)
        values_neg_interactions= np.zeros((len(df), len(columns_neg_interaction)), dtype=np.int8)
        current_index = {}
        i = 0
        for ncodpers in df.ncodpers:
            if not ncodpers in current_index:
                current_index[ncodpers] = 0
            c_idx = current_index[ncodpers]
            current_index[ncodpers] = current_index[ncodpers] + 1
            values_b_interactions[i] = b_interactions[ncodpers][c_idx]
            values_pos_interactions[i] = pos_interactions[ncodpers][c_idx]
            values_neg_interactions[i] = neg_interactions[ncodpers][c_idx]
            if(i % 2000000 == 0):
                print(i)
            i = i + 1
        df['b_interaction'] = values_b_interactions.astype(np.int8)
        df[columns_pos_interaction] = values_pos_interactions.astype(np.int8)
        for c in columns_pos_interaction:
            df[c] = df[c].astype(np.int8)
        df[columns_neg_interaction] = values_neg_interactions.astype(np.int8)
        for c in columns_neg_interaction:
            df[c] = df[c].astype(np.int8)

        
    else:
        print('Interactions already created')
        
    return df
        
        
        
def load_train_from_pickle_interactions():
    with open('pickles/train_santander_interactions.pickle', 'rb') as handle:
        df = pickle.load(handle)
    return df
    
def save_train_pickle_interactions(df):
    with open('pickles/train_santander_interactions.pickle', 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def load_aux_features_df(aux_features):
    with open('pickles/train_santander_aux_features.pickle', 'rb') as handle:
        df_aux_features = pickle.load(handle)

    for c in df_aux_features:
        if c not in aux_features:
            df_aux_features.drop(c, axis=1, inplace=True)
        
    return df_aux_features

	
	
'''
Build representation 1: State of products at every timestep
'''
def build_rep_1(df, df_test, max_interactions, aux_features):

    dtype_sparse = np.int8
    if len(aux_features) > 0:
        df_aux_features = load_aux_features_df(aux_features)
        df = df.join(df_aux_features, rsuffix='_r', lsuffix='_l')
        dtype_sparse = np.float32
    
    seq_length = max_interactions
    print('Building representation 1')
    #Build Training Matrix
    X_train = []
    Y_train = []
    
    grouped = df.groupby('ncodpers')
    prob_discard = 0.85
    val_date = df['fecha_dato'].unique()[-1]
    a = 0
    
    if False:
        with open('pickles/X_train_temp.pickle', 'rb') as handle:
            X_train = pickle.load(handle)
        with open('pickles/Y_train_temp.pickle', 'rb') as handle:
            Y_train = pickle.load(handle)
    else:
        for ncodpers,group in grouped:
            #if ncodpers == 1048484:
            for i in range(len(group)):
                if (i-seq_length) >= 0:
                    #print('train in index ' + str(i) + ' fecha ' + str(group.iloc[i]['fecha_dato']))
                    x = group[target_columns + aux_features].values[(i-seq_length):i,:]
                    y = group[target_columns].values[i,:]
                    if False in (x[-1][0:len(target_columns)] == y): #if the products doesn't change in last time step, save train sample, if not discard it with some probaiblity 
                        X_train.append(sparse.csr_matrix(x, dtype=dtype_sparse))
                        Y_train.append(sparse.csr_matrix(y, dtype=dtype_sparse))
                    elif np.random.rand() > prob_discard:
                        X_train.append(sparse.csr_matrix(x, dtype=dtype_sparse))
                        Y_train.append(sparse.csr_matrix(y, dtype=dtype_sparse))                   
                #else:
                    #print('Not enough data points past in index ' + str(i) + ' fecha ' + str(group.iloc[i]['fecha_dato']))
            a = a + 1
            if a % 100000 == 0:
                print(a)
            #if a % 350000 == 0:
                #break
            
        with open('pickles/X_train_temp.pickle', 'wb') as handle:
            pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('pickles/Y_train_temp.pickle', 'wb') as handle:
            pickle.dump(Y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
        
    diff_idx = []
    same_idx = []
    for i in range(len(X_train)):
        if False in (X_train[i].toarray()[-1, 0:len(target_columns)] == Y_train[i]):
            diff_idx.append(i)
        else:
            same_idx.append(i)
    print('Fnal train size: X' + str(len(X_train)))
    print('Fnal train size: Y' + str(len(Y_train)))
    print('same products: ' + str(len(same_idx)))
    print('Samples with changes in the last timestamp: ' + str(len(diff_idx)))
    
        
    #Build test matrix
    print('Build test set matrix')
    ncodpers_test = df_test['ncodpers']
    i = 0
    print(len(ncodpers_test))
    X_test = np.zeros((len(ncodpers_test), seq_length, len(target_columns) + len(aux_features)), dtype=np.float32)
    for ncodpers in ncodpers_test:
        #if ncodpers == 1048484:
        x = df[df['ncodpers']== ncodpers][target_columns + aux_features].values[(-seq_length):]
        if len(x) < seq_length:
            result = np.zeros((seq_length, len(target_columns) + len(aux_features)))
            result[(seq_length - len(x)):,:] = x #Test the other way
            x = result
        #X_test.append(x)
        X_test[i] = x
        i = i + 1
        if i % 10000 == 0:
            print(i)
            
    #Build local test matrix
    print('Build local test set matrix')
    with open('pickles/ncodpers_interactions_local_test.pickle', 'rb') as handle:
        ncodpers_interactions_local_test = pickle.load(handle)
    X_local_test = []
    Y_local_test = []
    local_test_date = df.fecha_dato.max()
    grouped = df[df.ncodpers.isin(ncodpers_interactions_local_test)].groupby('ncodpers')
    i = 0
    for name,group in grouped:
        last_date = group.fecha_dato.max()
        if last_date == local_test_date:
            last_values = group[target_columns + aux_features].values[(-(seq_length+1)):]
            x = last_values[0:len(last_values)-1]
            y = last_values[-2:]
            y = y[1] - y[0] #What if recently added and not enough y? Discard  that ncodpers?
            y = (y > 0).astype(np.int8)
            if len(x) < seq_length:
                result = np.zeros((seq_length, len(target_columns) + len(aux_features)))
                result[(seq_length - len(x)):,:] = x #Test the other way
                x = result
            X_local_test.append(x)
            Y_local_test.append(y)
        i = i + 1
        if i % 10000 == 0:
            print(i)
            
    with open('pickles/X_test_temp.pickle', 'wb') as handle:
        pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return X_train, Y_train, X_test, X_local_test, Y_local_test
    
    
'''
Build representation 2: Same representation 1, but we predict the adds instead of the state
'''
def build_rep_2(df, df_test, max_interactions, aux_features):

    columns_pos_interaction = []
    for prod in target_columns:
        columns_pos_interaction.append(prod + '_pos_interaction')

    dtype_sparse = np.int8
    if len(aux_features) > 0:
        df_aux_features = load_aux_features_df(aux_features)
        df = df.join(df_aux_features, rsuffix='_r', lsuffix='_l')
        dtype_sparse = np.float32
    
    seq_length = max_interactions
    print('Building representation 2')
    #Build Training Matrix
    X_train = []
    Y_train = []
    
    local_test_date = df.fecha_dato.max()
    grouped = df[df.fecha_dato != local_test_date].groupby('ncodpers')
    #val_date = df['fecha_dato'].unique()[-1]
    a = 0
    

    for ncodpers,group in grouped:
        #if ncodpers == 1048484:
        for i in range(len(group)):
            if (i-seq_length) >= 0:
                #print('train in index ' + str(i) + ' fecha ' + str(group.iloc[i]['fecha_dato']))
                x = group[target_columns + aux_features].values[(i-seq_length):i,:]
                y = group[columns_pos_interaction].values[i,:]
                if 1 in y: #if the products doesn't change in last time step, save train sample, if not discard it with some probaiblity 
                    X_train.append(sparse.csr_matrix(x, dtype=dtype_sparse))
                    Y_train.append(sparse.csr_matrix(y, dtype=dtype_sparse))
        a = a + 1
        if a % 100000 == 0:
            print(a)
        #if a % 350000 == 0:
            #break
        
    with open('pickles/X_train_temp.pickle', 'wb') as handle:
        pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickles/Y_train_temp.pickle', 'wb') as handle:
        pickle.dump(Y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
        

    print('Fnal train size: X' + str(len(X_train)))
    print('Fnal train size: Y' + str(len(Y_train)))

    
        
    #Build test matrix
    print('Build test set matrix')
    ncodpers_test = df_test['ncodpers']
    i = 0
    print(len(ncodpers_test))
    X_test = np.zeros((len(ncodpers_test), seq_length, len(target_columns) + len(aux_features)), dtype=np.float32)
    for ncodpers in ncodpers_test:
        #if ncodpers == 1048484:
        x = df[df['ncodpers']== ncodpers][target_columns + aux_features].values[(-seq_length):]
        if len(x) < seq_length:
            result = np.zeros((seq_length, len(target_columns) + len(aux_features)))
            result[(seq_length - len(x)):,:] = x #Test the other way
            x = result
        #X_test.append(x)
        X_test[i] = x
        i = i + 1
        if i % 10000 == 0:
            print(i)
            
    #Build local test matrix
    print('Build local test set matrix')
    with open('pickles/ncodpers_interactions_local_test.pickle', 'rb') as handle:
        ncodpers_interactions_local_test = pickle.load(handle)
    X_local_test = []
    Y_local_test = []
    grouped = df[df.ncodpers.isin(ncodpers_interactions_local_test)].groupby('ncodpers')
    i = 0
    for name,group in grouped:
        last_date = group.fecha_dato.max()
        if last_date == local_test_date:
            #last_values = group[target_columns + aux_features].values[(-(seq_length+1)):]
            #x = last_values[0:len(last_values)-2]
            x = group[group.fecha_dato < local_test_date][target_columns + aux_features].values[-seq_length:]
            y = group[group.fecha_dato == local_test_date][columns_pos_interaction].values.astype(np.int8)

            if len(x) < seq_length:
                result = np.zeros((seq_length, len(target_columns) + len(aux_features)))
                result[(seq_length - len(x)):,:] = x #Test the other way
                x = result
            X_local_test.append(x)
            Y_local_test.append(y)
        i = i + 1
        if i % 10000 == 0:
            print(i)
            
    with open('pickles/X_test_temp.pickle', 'wb') as handle:
        pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return X_train, Y_train, X_test, X_local_test, Y_local_test
    
    
    
'''
Build representation 4: Only vectors of interactions formed by: Positive interactions(adds) + Negative interactions(drops) + time from last interaction
'''
def build_rep_4(df, df_test, max_interactions, aux_features, padding): 

    if False:
        aux_features_length = str(len(aux_features))
        with open('pickles/X_train_temp.pickle', 'rb') as handle:
            X_train = pickle.load(handle)
        with open('pickles/X_test_temp.pickle', 'rb') as handle:
            X_test = pickle.load(handle)
        with open('pickles/Y_train_temp.pickle', 'rb') as handle:
            Y_train = pickle.load(handle)
        with open('pickles/X_local_test_temp.pickle', 'rb') as handle:
            X_local_test = pickle.load(handle)
        with open('pickles/Y_local_test_temp.pickle', 'rb') as handle:
            Y_local_test = pickle.load(handle)
        
    else:

        dtype_sparse = np.int8
        if len(aux_features) > 0:
            df_aux_features = load_aux_features_df(aux_features)
            df = df.join(df_aux_features, rsuffix='_r', lsuffix='_l')
            dtype_sparse = np.float32
        print('Building representation 4')
        time_column = 'time_from_last_interaction'
        X_train = []
        Y_train = []
        X_test = []
        columns_pos_interaction = []
        columns_neg_interaction = []
        for prod in target_columns:
            columns_pos_interaction.append(prod + '_pos_interaction')
            columns_neg_interaction.append(prod + '_neg_interaction')
        a = 0
        ncodpers_test = df_test['ncodpers'].tolist()
        X_test = np.zeros((len(ncodpers_test), max_interactions, len(columns_pos_interaction) + len(columns_pos_interaction) + 1 + len(aux_features)), dtype=np.int8)
        grouped = df.groupby('ncodpers')
        for ncodpers,group in grouped:
            
            interactions = group[columns_pos_interaction + columns_neg_interaction + [time_column] + aux_features]
            #add_interactions = group[columns_pos_interaction + [time_column]]
            interactions = interactions[group['b_interaction'] == True].values
            num_interactions = len(interactions)
            for i in range(1, num_interactions):
                y = interactions[i][0:len(columns_pos_interaction)] 
                if 1 in y: #Only add samples with added products
                    x = np.zeros((max_interactions, interactions.shape[1]))
                    interactions_sample = interactions[:i, :]
                    if len(interactions_sample) > 0:
                        if len(interactions_sample > max_interactions):
                            interactions_sample = interactions_sample[-max_interactions:, :]
                        if padding.lower() == 'right':
                            x[:len(interactions_sample), : ] = interactions_sample #Padding Right
                        elif padding.lower() == 'left':
                            x[-len(interactions_sample):, :] = interactions_sample #Padding Left
                    X_train.append(sparse.csr_matrix(x, dtype=dtype_sparse))
                    Y_train.append(sparse.csr_matrix(y, dtype=dtype_sparse))
            if ncodpers in ncodpers_test:
                idx_test = ncodpers_test.index(ncodpers)
                x_test = np.zeros((max_interactions, interactions.shape[1]))
                if num_interactions > 0:
                    if len(interactions) > max_interactions:
                        interactions = interactions[-max_interactions:, :]
                    if padding.lower() == 'right':
                        x_test[:len(interactions):, :] = interactions #Padding Right
                    elif padding.lower() == 'left':
                        x_test[-len(interactions):, :] = interactions #Padding Left
                X_test[idx_test] = x_test
            if a % 10000 == 0:
                print(a)
                
            a = a + 1
            
            
        #Build local test matrix
        print('Build local test set matrix')
        with open('pickles/ncodpers_interactions_local_test.pickle', 'rb') as handle:
            ncodpers_interactions_local_test = pickle.load(handle)
        X_local_test = []
        Y_local_test = []
        local_test_date = df.fecha_dato.max()
        grouped = df[df.ncodpers.isin(ncodpers_interactions_local_test)].groupby('ncodpers')
        i = 0
        for name,group in grouped:
            last_date = group.fecha_dato.max()
            if last_date == local_test_date:
                interactions = group[group.fecha_dato < local_test_date][columns_pos_interaction + columns_neg_interaction + [time_column] + aux_features]
                #add_interactions = group[columns_pos_interaction + [time_column]]
                interactions = interactions[group['b_interaction'] == True].values
                num_interactions = len(interactions)
                x_test = np.zeros((max_interactions, interactions.shape[1]), dtype=np.int8)
                if num_interactions > 0:
                    if len(interactions) > max_interactions:
                        interactions = interactions[-max_interactions:, :]
                    if padding.lower() == 'right':
                        x_test[:len(interactions):, :] = interactions #Padding Right
                    elif padding.lower() == 'left':
                        x_test[-len(interactions):, :] = interactions #Padding Left
                
                y_test = group[group.fecha_dato == local_test_date][columns_pos_interaction].values.astype(np.int8)            
                    
                X_local_test.append(x_test)
                Y_local_test.append(y_test)
            i = i + 1
            if i % 10000 == 0:
                print(i)
            
        with open('pickles/X_train_temp.pickle', 'wb') as handle:
            pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('pickles/Y_train_temp.pickle', 'wb') as handle:
            pickle.dump(Y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('pickles/X_test_temp.pickle', 'wb') as handle:
            pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('pickles/X_local_test_temp.pickle', 'wb') as handle:
            pickle.dump(X_local_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('pickles/Y_local_test_temp.pickle', 'wb') as handle:
            pickle.dump(Y_local_test, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    
    return X_train, Y_train, X_test, X_local_test, Y_local_test
    
 