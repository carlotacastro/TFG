def split_func (data, X, y, end_date, test_size):
    
    # Splitting train and test
    idx_train, idx_test = train_test_split(data.index, test_size=test_size, shuffle=False)
    X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]
    
    return X_train, y_train, X_test, y_test
    
    

    
    
  def train_test (data, end_df, n):
    
    df = data.loc[:end_df,:].reset_index().set_index(['store_nbr', 'family', 'date']).sort_index()
    y = np.log1p(df.loc[:,'sales'].unstack(['store_nbr', 'family']))
    
    # Selecting features
    #Eliminem features que no ens aporten res com els estats, l'id o el tipus de botiga i no tenim en compte 'onpromotion' ja 
    #que dona molts de problemes amb valors NanS
    X = df.drop(columns = ['id', 'sales', 'transactions', 'onpromotion', 'store_B', 'store_C', 'store_D', 'store_E', 'state'])

    X = X.groupby(by='date').first()
        
    # Train
    if end_df <= date['date_end_train']:
        y_tr = np.empty((diff_train-n,0))
        y_te = np.empty((n,0))
        pred_train_y = np.empty((diff_train-n,0))
        pred_test_y = np.empty((n,0))
    # Test
    else:
        y_tr = np.empty((diff_test-n,0))
        y_te = np.empty((n,0))
        pred_train_y = np.empty((diff_test-n,0))
        pred_test_y = np.empty((n,0))

    # A model for each shop
    for i in data.store_nbr.unique():
        y = df.loc[i,'sales'].unstack(['family'])
        X = df.loc[i,X.columns]
        X = X.groupby(by='date').first()

        # Splitting train and test and log transformation
        X_train, y_train, X_test, y_test = split_func(y, X, np.log1p(y), end_df, n)
                
        # Exponentially weighted cost function
        weights = X_train.year_2017.apply(lambda x: np.exp((-0.9)*1) if x == 0 else np.exp((-0.9)*0)) 
        
        # XGBRegressor
        model = xg.XGBRegressor(n_estimators=500, learning_rate = 0.01, max_depth= 3, subsample = 0.5, colsample_bytree = 0.4, colsample_bylevel = 1, random_state=0)
        model.fit(X_train, y_train, sample_weight=weights)
        xg_pred_train_y = model.predict(X_train) 
        xg_pred_test_y = model.predict(X_test)
        
        y_tr = np.append(y_tr, y_train, axis=1)
        y_te = np.append(y_te, y_test, axis=1)
        pred_train_y = np.append(pred_train_y, xg_pred_train_y, axis=1)
        pred_test_y = np.append(pred_test_y, xg_pred_test_y, axis=1)
        
        # Performances of each shop
        # Train
        if end_df <= date['date_end_train']:
            print(f'RMSLE_train {i}: ', np.round(np.sqrt(mean_squared_error(y_train.clip(0.0), xg_pred_train_y.clip(0.0))), 4), f'RMSLE_test {i}: ', np.round(np.sqrt(mean_squared_error(y_test.clip(0.0), xg_pred_test_y.clip(0.0))), 4))

    index = pd.MultiIndex.from_product([data.store_nbr.unique(), data.family.sort_values().unique()], names=['store_nbr', 'family'])
    
    y_tr = pd.DataFrame(y_tr, columns=index, index=X_train.index)
    y_te = pd.DataFrame(y_te, columns=index, index=X_test.index)
    pred_train_y = pd.DataFrame(pred_train_y, columns=y_tr.columns, index=y_tr.index)
    pred_test_y = pd.DataFrame(pred_test_y, columns=y_te.columns, index=y_te.index)
    
    # Total performances
    # Train
    if end_df <= date['date_end_train']:
        print(f'RMSLE_train tot: ', np.round(np.sqrt(mean_squared_error(y_tr.clip(0.0), pred_train_y.clip(0.0))), 4), f'RMSLE_test tot: ', np.round(np.sqrt(mean_squared_error(y_te.clip(0.0), pred_test_y.clip(0.0))), 4))

   
    y_tr = y_tr.stack(['store_nbr', 'family'])
    y_te = y_te.stack(['store_nbr', 'family'])
    #y = pd.concat([y_tr, y_te])
    
    pred_train_y = pred_train_y.stack(['store_nbr', 'family'])
    pred_test_y = pred_test_y.stack(['store_nbr', 'family'])
    # If sales < 0 -> 0
    #pred_y = pd.concat([pd.Series(pred_train_y).apply(lambda x: 0 if x<0 else x), pd.Series(pred_test_y).apply(lambda x: 0 if x<0 else x)])
    
    return pred_test_y, y_te
  
  
  
  
  def HT (data, end_df, n):
    
    df = data.loc[:end_df,:].reset_index().set_index(['store_nbr', 'family', 'date']).sort_index()
    y = np.log1p(df.loc[:,'sales'].unstack(['store_nbr', 'family']))
    
    # Selecting features
    #Eliminem features que no ens aporten res com els estats, l'id o el tipus de botiga i no tenim en compte 'onpromotion' ja 
    #que dona molts de problemes amb valors NanS
    X = df.drop(columns = ['id', 'sales', 'onpromotion', 'store_B', 'store_C', 'store_D', 'store_E', 'state'])
    X = X.groupby(by='date').first()
        
    # Train
    if end_df <= date['date_end_train']:
        y_tr = np.empty((diff_train-n,0))
        y_te = np.empty((n,0))
        pred_train_y = np.empty((diff_train-n,0))
        pred_test_y = np.empty((n,0))
    # Test
    else:
        y_tr = np.empty((diff_test-n,0))
        y_te = np.empty((n,0))
        pred_train_y = np.empty((diff_test-n,0))
        pred_test_y = np.empty((n,0))

    # A model for each shop
    for i in data.store_nbr.unique():
        y = df.loc[i,'sales'].unstack(['family'])
        X = df.loc[i,X.columns]
        X = X.groupby(by='date').first()

        # Splitting train and test and log transformation
        X_train, y_train, X_test, y_test = split_func(y, X, np.log1p(y), end_df, n)
                
        # Exponentially weighted cost function
        weights = X_train.year_2017.apply(lambda x: np.exp((-0.9)*1) if x == 0 else np.exp((-0.9)*0)) 
        
        #Tuning
        # Number of trees in random forest
        params = { 'max_depth': [3, 5, 6, 10, 15, 20],
                  'learning_rate': [0.01, 0.1, 0.2, 0.3],
                  'subsample': np.arange(0.5, 1.0, 0.1),
                  'colsample_bytree': np.arange(0.4, 1.0, 0.1),
                  'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
                  'n_estimators': [100, 500, 1000]}
        
        # XGBRegressor
        model = xg.XGBRegressor(n_estimators=320, random_state=0)
        random_search = RandomizedSearchCV(model, param_distributions=params, scoring='neg_mean_absolute_error', cv=3, random_state=42, n_jobs=-1)
        random_search.fit(X_train, y_train, sample_weight=weights)
        xg_pred_train_y = random_search.predict(X_train) 
        xg_pred_test_y = random_search.predict(X_test)
        
        y_tr = np.append(y_tr, y_train, axis=1)
        y_te = np.append(y_te, y_test, axis=1)
        pred_train_y = np.append(pred_train_y, xg_pred_train_y, axis=1)
        pred_test_y = np.append(pred_test_y, xg_pred_test_y, axis=1)
        
        # Performances of each shop
        # Train
        if end_df <= date['date_end_train']:
            print(f'RMSLE_train {i}: ', np.round(np.sqrt(mean_squared_error(y_train.clip(0.0), xg_pred_train_y.clip(0.0))), 4), f'RMSLE_test {i}: ', np.round(np.sqrt(mean_squared_error(y_test.clip(0.0), xg_pred_test_y.clip(0.0))), 4))
        

    index = pd.MultiIndex.from_product([data.store_nbr.unique(), data.family.sort_values().unique()], names=['store_nbr', 'family'])
    
    y_tr = pd.DataFrame(y_tr, columns=index, index=X_train.index)
    y_te = pd.DataFrame(y_te, columns=index, index=X_test.index)
    pred_train_y = pd.DataFrame(pred_train_y, columns=y_tr.columns, index=y_tr.index)
    pred_test_y = pd.DataFrame(pred_test_y, columns=y_te.columns, index=y_te.index)
    
    # Total performances
    # Train
    if end_df <= date['date_end_train']:
        print(f'RMSLE_train tot: ', np.round(np.sqrt(mean_squared_error(y_tr.clip(0.0), pred_train_y.clip(0.0))), 4), f'RMSLE_test tot: ', np.round(np.sqrt(mean_squared_error(y_te.clip(0.0), pred_test_y.clip(0.0))), 4))

   
    y_tr = y_tr.stack(['store_nbr', 'family'])
    y_te = y_te.stack(['store_nbr', 'family'])
   # y = pd.concat([y_tr, y_te])
    
    pred_train_y = pred_train_y.stack(['store_nbr', 'family'])
    pred_test_y = pred_test_y.stack(['store_nbr', 'family'])
    # If sales < 0 -> 0
  #  pred_y = pd.concat([pd.Series(pred_train_y).apply(lambda x: 0 if x<0 else x), pd.Series(pred_test_y).apply(lambda x: 0 if x<0 else x)])
    
    return pred_test_y, df
