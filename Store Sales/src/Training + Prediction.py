def split_func (data, X, y, end_date, test_size):
    
    # Splitting train and test
    idx_train, idx_test = train_test_split(data.index, test_size=test_size, shuffle=False)
    X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]
    
    return X_train, y_train, X_test, y_test
    
    
def create_sample_weights(X, target_date, weight=0.9):
    extra_weight_days = X.index.get_level_values('date') > target_date
    return np.array(list(map(lambda x: np.exp(-weight) if x == 0 else 1, extra_weight_days.astype('int'))))

    
    
def train_test (data, end_df, n):
    
    df = data.loc[:end_df,:].reset_index().set_index(['store_nbr', 'family', 'date']).sort_index()
    y = np.log1p(df.loc[:,'sales'].unstack(['store_nbr', 'family']))
    
    # Selecting features
    #We select the best feature (done in feature selection)
    
    X = df[['sin(2,freq=A-DEC)','sin(2,freq=W-SUN)','lagoil_2_dcoilwtico','lagoil_6_dcoilwtico',
            'lagoil_7_dcoilwtico','isweekend','oil_2_month_avg','trend','lagoil_10_dcoilwtico',
            'sin(1,freq=A-DEC)','lagoil_1_dcoilwtico','sin(4,freq=W-SUN)','cos(1,freq=W-SUN)',
            'dcoilwtico','sin(4,freq=A-DEC)','oil_1_month_avg','lagoil_14_dcoilwtico',
            'sin(5,freq=A-DEC)','sin(1,freq=M)','cos(2,freq=M)','day','cos(1,freq=M)','sin(2,freq=M)',
            'sin(1,freq=W-SUN)','onpromotion_std_store', 'onpromotion_avg_store', 'onpromotion_biweek_avg',
            'onpromotion_lag_3'
            ]] 
    
    X = X.groupby(by='date').first()
        
    # Train
    if end_df <= date['date_end_train']:
        y_tr = np.empty((92,0))
        y_te = np.empty((n,0))
        pred_train = np.empty((92,0))
        pred_test = np.empty((n,0))
    # Test
    else:
        y_tr = np.empty((108,0))
        y_te = np.empty((n,0))
        pred_train = np.empty((108,0))
        pred_test = np.empty((n,0))

    # A model for each shop
    for i in data.store_nbr.unique():
        y = df.loc[i,'sales'].unstack(['family'])
        X = df.loc[i, X.columns]
        X = X.groupby(by='date').first()

        # Splitting train and test and log transformation
        X_train, y_train, X_test, y_test = split_func(y, X, np.log1p(y), end_df, n)
                
        # Exponentially weighted cost function
        weights = create_sample_weights(X_train, '2017-07-01')
        
        # RandomForestRegressor
        model = RandomForestRegressor(n_estimators=1200, max_depth = 50, max_features = 'auto', bootstrap = True, min_samples_leaf=2, min_samples_split=2, random_state=0) #bootstrap=False, max_depth=90, max_features='sqrt',min_samples_leaf=4, min_samples_split=10,n_estimators=600
        model.fit(X_train, y_train, sample_weight = weights) 
        rf_pred_train = model.predict(X_train) 
        rf_pred_test = model.predict(X_test)
        
        # XGBRegressor
        model = xg.XGBRegressor(n_estimators=500, learning_rate = 0.01, max_depth= 15, subsample = 0.5, colsample_bytree = 0.4, colsample_bylevel = 1, random_state=0)
        model.fit(X_train, y_train, sample_weight=weights)
        xg_pred_train = model.predict(X_train) 
        xg_pred_test = model.predict(X_test)
        
        # Average result
        total_pred_train = xg_pred_train * 0.3 + rf_pred_train * 0.7
        total_pred_test = xg_pred_test * 0.3 + rf_pred_test * 0.7
        
        y_tr = np.append(y_tr, y_train, axis=1)
        y_te = np.append(y_te, y_test, axis=1)
        pred_train = np.append(pred_train, total_pred_train, axis=1)
        pred_test = np.append(pred_test, total_pred_test, axis=1)
        
        # Performances of each shop
        # Train
        if end_df <= date['date_end_train']:
            print(f'RMSLE_train {i}: ', np.round(np.sqrt(mean_squared_error(y_train.clip(0.0), total_pred_train.clip(0.0))), 4), f'RMSLE_test {i}: ', np.round(np.sqrt(mean_squared_error(y_test.clip(0.0), total_pred_test.clip(0.0))), 4))        

    index = pd.MultiIndex.from_product([data.store_nbr.unique(), data.family.sort_values().unique()], names=['store_nbr', 'family'])
    
    y_tr = pd.DataFrame(y_tr, columns=index, index=X_train.index)
    y_te = pd.DataFrame(y_te, columns=index, index=X_test.index)
    pred_train = pd.DataFrame(pred_train, columns=y_tr.columns, index=y_tr.index)
    pred_test = pd.DataFrame(pred_test, columns=y_te.columns, index=y_te.index)
    
    # Total performances
    # Train
    if end_df <= date['date_end_train']:
        print(f'RMSLE_train tot: ', np.round(np.sqrt(mean_squared_error(y_tr.clip(0.0), pred_train.clip(0.0))), 4), f'RMSLE_test tot: ', np.round(np.sqrt(mean_squared_error(y_te.clip(0.0), pred_test.clip(0.0))), 4))

   
    y_tr = y_tr.stack(['store_nbr', 'family'])
    y_te = y_te.stack(['store_nbr', 'family'])
    pred_train = pred_train.stack(['store_nbr', 'family'])
    pred_test = pred_test.stack(['store_nbr', 'family'])
 
    return pred_test, y_te
  
  
  
  
def HT (data, end_df, n):
    
    df = features(data).loc[:end_df,:].reset_index().set_index(['store_nbr', 'family', 'date']).sort_index()
    y = np.log1p(df.loc[:,'sales'].unstack(['store_nbr', 'family']))
    
    # Selecting features
    X = df[['sin(2,freq=A-DEC)','sin(2,freq=W-SUN)','lagoil_2_dcoilwtico','lagoil_6_dcoilwtico',
            'lagoil_7_dcoilwtico','isweekend','oil_2_month_avg','trend','lagoil_10_dcoilwtico',
            'sin(1,freq=A-DEC)','lagoil_1_dcoilwtico','sin(4,freq=W-SUN)','cos(1,freq=W-SUN)',
            'dcoilwtico','sin(4,freq=A-DEC)','oil_1_month_avg','lagoil_14_dcoilwtico',
            'sin(5,freq=A-DEC)','sin(1,freq=M)','cos(2,freq=M)','day','cos(1,freq=M)','sin(2,freq=M)',
            'sin(1,freq=W-SUN)','onpromotion_std_store', 'onpromotion_avg_store', 'onpromotion_biweek_avg'
            ]] 
    
    X = X.groupby(by='date').first()
        
    # Train
    if end_df <= date['date_end_train']:
        y_tr = np.empty((92,0))
        y_te = np.empty((n,0))
        pred_train = np.empty((92,0))
        pred_test = np.empty((n,0))
    # Test
    else:
        y_tr = np.empty((diff_test-n,0))
        y_te = np.empty((n,0))
        pred_train = np.empty((diff_test-n,0))
        pred_test = np.empty((n,0))
        
    params = []

    # A model for each shop
    for i in data.store_nbr.unique():
        y = df.loc[i,'sales'].unstack(['family'])
        X = df.loc[i,X.columns]
        X = X.groupby(by='date').first()

        # Splitting train and test and log transformation
        X_train, y_train, X_test, y_test = split_func(y, X, np.log1p(y), end_df, n)
                
        # Exponentially weighted cost function    
        weights = create_sample_weights(X_train, '2017-07-01')
        
        random_grid = { 'max_depth': [3, 5, 6, 10, 15, 20],
                  'learning_rate': [0.01, 0.1, 0.2, 0.3],
                  'subsample': np.arange(0.5, 1.0, 0.1),
                  'colsample_bytree': np.arange(0.4, 1.0, 0.1),
                  'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
                  'n_estimators': [100, 500, 1000]}


        # Random Forest
        model = xg.XGBRegressor(random_state=0)
        random_search = RandomizedSearchCV(model, param_distributions=random_grid, scoring='neg_mean_absolute_error', cv=3, random_state=42, n_jobs=-1)
        random_search.fit(X_train, y_train, sample_weight = weights)
        params.append(random_search.best_params_)
        rf_pred_train = random_search.predict(X_train) 
        rf_pred_test = random_search.predict(X_test)
        
        y_tr = np.append(y_tr, y_train, axis=1)
        y_te = np.append(y_te, y_test, axis=1)
        pred_train = np.append(pred_train, rf_pred_train, axis=1)
        pred_test = np.append(pred_test, rf_pred_test, axis=1)
        
        # Performances of each shop
        # Train
        if end_df <= date['date_end_train']:
            print(f'RMSLE_train {i}: ', np.round(np.sqrt(mean_squared_error(y_train.clip(0.0), rf_pred_train.clip(0.0))), 4), f'RMSLE_test {i}: ', np.round(np.sqrt(mean_squared_error(y_test.clip(0.0), rf_pred_test.clip(0.0))), 4))
        

    index = pd.MultiIndex.from_product([data.store_nbr.unique(), data.family.sort_values().unique()], names=['store_nbr', 'family'])
    
    y_tr = pd.DataFrame(y_tr, columns=index, index=X_train.index)
    y_te = pd.DataFrame(y_te, columns=index, index=X_test.index)
    pred_train = pd.DataFrame(pred_train, columns=y_tr.columns, index=y_tr.index)
    pred_test = pd.DataFrame(pred_test, columns=y_te.columns, index=y_te.index)
    
    # Total performances
    # Train
    if end_df <= date['date_end_train']:
        print(f'RMSLE_train tot: ', np.round(np.sqrt(mean_squared_error(y_tr.clip(0.0), pred_train.clip(0.0))), 4), f'RMSLE_test tot: ', np.round(np.sqrt(mean_squared_error(y_te.clip(0.0), pred_test.clip(0.0))), 4))

   
    y_tr = y_tr.stack(['store_nbr', 'family'])
    y_te = y_te.stack(['store_nbr', 'family'])
    pred_train = pred_train.stack(['store_nbr', 'family'])
    pred_test = pred_test.stack(['store_nbr', 'family'])

    return  pred_test, y_te, params, random_search
