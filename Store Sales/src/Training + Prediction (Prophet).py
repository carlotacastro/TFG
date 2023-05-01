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
    X = df.drop(columns = ['id', 'type','sales','trend','transactions', 'onpromotion', 'store_B', 'store_C', 'store_D', 'store_E'])
    df2 = X.groupby(by='date').first()
        
    # Train
    if end_df <= date['date_end_train']:
        y_tr = np.empty((3036,0))
        y_te = np.empty((528,0))
        pred_train_y = np.empty((3036,0))
        pred_test_y = np.empty((528,0))
    # Test
    else:
        y_tr = np.empty((3564,0))
        y_te = np.empty((528,0))
        pred_train_y = np.empty((3564,0))
        pred_test_y = np.empty((528,0))

    # A model for each shop
    for i in data.store_nbr.unique():
        y = df.loc[i,'sales'].unstack(['family'])
        X = df.loc[i, df2.columns]
        X = X.groupby(by='date').first()

        # Splitting train and test and log transformation
        X_train, y_train, X_test, y_test = split_func(y, X, np.log1p(y), end_df, n)
        
        y_train = y_train.stack(['family']).to_frame()
        
        if end_df > date['date_end_train']:
            y_test = y_test.fillna(0).stack(['family']).to_frame()
        y_test = y_test.stack(['family']).to_frame()

        train = y_train.join(X_train.reindex(y_train.index, level=0))
        train = train.reset_index()
        train = train.rename(columns={'date': 'ds', 0: 'y'})

        test = y_test.join(X_test.reindex(y_test.index, level=0))
        test = test.reset_index()
        test = test.rename(columns={'date': 'ds', 0: 'y'})
        if end_df > date['date_end_train']:
            test['y'] = np.nan

        y_train = y_train.reset_index()
        y_test = y_test.reset_index()
        y_train = y_train[['date',0]].rename(columns={'date': 'ds', 0: 'y'})
        y_test = y_test[['date',0]].rename(columns={'date': 'ds', 0: 'y'})
    

        # Prophet
        model = Prophet(holidays = event_holiday,
                        changepoint_prior_scale = 0.05,
                        holidays_prior_scale = 0.01,
                        seasonality_prior_scale = 0.01,
                        seasonality_mode = 'additive',
                        yearly_seasonality = False,
                          weekly_seasonality = True,
                          daily_seasonality = False)
        
        for j in range(0, len(df2.columns.values)):
            model.add_regressor(df2.columns.values[j])
            
        model.fit(train)
        p_pred_train_y = model.predict(train) 
        p_pred_test_y = model.predict(test)
        
        y_tr = np.append(y_tr, train[['y']], axis=1)
        y_te = np.append(y_te, test[['y']], axis=1)
        pred_train_y = np.append(pred_train_y, p_pred_train_y[['yhat']], axis=1)
        pred_test_y = np.append(pred_test_y, p_pred_test_y[['yhat']], axis=1)
        
        # Performances of each shop
        # Train
        if end_df <= date['date_end_train']:
            print(f'RMSLE_train {i}: ', np.round(np.sqrt(mean_squared_error(train[['y']].clip(0.0), p_pred_train_y[['ds', 'yhat']].set_index('ds').clip(0.0))), 4), f'RMSLE_test {i}: ', np.round(np.sqrt(mean_squared_error(test[['y']].clip(0.0), p_pred_test_y[['ds', 'yhat']].set_index('ds').clip(0.0))), 4))        

    
    # Total performances
    # Train
    if end_df <= date['date_end_train']:
        print(f'RMSLE_train tot: ', np.round(np.sqrt(mean_squared_error(y_tr.clip(0.0), pred_train_y.clip(0.0))), 4), f'RMSLE_test tot: ', np.round(np.sqrt(mean_squared_error(y_te.clip(0.0), pred_test_y.clip(0.0))), 4))

   
 
    return pred_test_y, y_te






def tuning(train, test, y_test, X_test):
    param_grid = {  
    'changepoint_prior_scale': [0.001, 0.05, 0.08, 0.5],
    'holidays_prior_scale': [0.01, 1, 5, 10, 12],
    'seasonality_prior_scale': [0.01, 1, 5, 10, 12]
    }
    grid = ParameterGrid(param_grid)
    
    model_parameters = pd.DataFrame(columns = ['RMSLE','Parameters'])
    for p in grid:
        random.seed(0)
        train_model = Prophet(changepoint_prior_scale = p['changepoint_prior_scale'],
                          holidays_prior_scale = p['holidays_prior_scale'],
                          seasonality_prior_scale = p['seasonality_prior_scale'],
                          yearly_seasonality = False,
                          weekly_seasonality = True,
                          daily_seasonality = False)
        
        print('He entrado', p)
        train_model.fit(train)
        p_pred_test_y = train_model.predict(test)
        RMSLE = np.round(np.sqrt(mean_squared_error(test[['y']].clip(0.0), p_pred_test_y[['ds', 'yhat']].set_index('ds').clip(0.0))), 4)
        model_parameters = model_parameters.append({'RMSLE':RMSLE,'Parameters':p},ignore_index=True)
        
    parameters = model_parameters.sort_values(by=['RMSLE'])
    parameters = parameters.reset_index(drop=True)
    return parameters['Parameters'][0]
  
  
  
  
  
  
  def HT (data, end_df, n):
    
    df = data.loc[:end_df,:].reset_index().set_index(['store_nbr', 'family', 'date']).sort_index()
    y = np.log1p(df.loc[:,'sales'].unstack(['store_nbr', 'family']))
    
    # Selecting features
    #Eliminem features que no ens aporten res com l'id o el tipus de botiga i no tenim en compte 'onpromotion' i transactions ja 
    #que dona molts de problemes amb valors NanS
    X = df.drop(columns = ['id', 'type','sales','trend','transactions', 'onpromotion', 'store_B', 'store_C', 'store_D', 'store_E'])
    df2 = X.groupby(by='date').first()
        
    # Train
    if end_df <= date['date_end_train']:
        y_tr = np.empty((3036,0))
        y_te = np.empty((528,0))
        pred_train_y = np.empty((3036,0))
        pred_test_y = np.empty((528,0))
    # Test
    else:
        y_tr = np.empty((3564,0))
        y_te = np.empty((528,0))
        pred_train_y = np.empty((3564,0))
        pred_test_y = np.empty((528,0))
        
    par = []
    
    # A model for each shop
    for i in data.store_nbr.unique():
        y = df.loc[i,'sales'].unstack(['family'])
        X = df.loc[i, df2.columns]
        X = X.groupby(by='date').first()

        # Splitting train and test and log transformation
        X_train, y_train, X_test, y_test = split_func(y, X, np.log1p(y), end_df, n)
        
        y_train = y_train.stack(['family']).to_frame()
        if end_df > date['date_end_train']:
            y_test = y_test.fillna(0).stack(['family']).to_frame()
        y_test = y_test.stack(['family']).to_frame()

        train = y_train.join(X_train.reindex(y_train.index, level=0))
        train = train.reset_index()
        train = train.rename(columns={'date': 'ds', 0: 'y'})

        test = y_test.join(X_test.reindex(y_test.index, level=0))
        test = test.reset_index()
        test = test.rename(columns={'date': 'ds', 0: 'y'})
        if end_df > date['date_end_train']:
            test['y'] = np.nan

        #         y_train = y_train.stack(['family']).to_frame()
        #         y_test = y_test.stack(['family']).to_frame()

        #         train = y_train.join(X_train.reindex(y_train.index, level=0))
        #         train = train.reset_index()
        #         train = train[['date',0]].rename(columns={'date': 'ds', 0: 'y'})

        #cambiar nombres de la date por ds y de las sales por y
        y_train = y_train.reset_index()
        y_test = y_test.reset_index()
        y_train = y_train[['date',0]].rename(columns={'date': 'ds', 0: 'y'})
        y_test = y_test[['date',0]].rename(columns={'date': 'ds', 0: 'y'})

        params = tuning(train, test, y_test, X_test)
        par.append(params)
        model = Prophet(changepoint_prior_scale=params['changepoint_prior_scale'], 
                     seasonality_prior_scale=params['seasonality_prior_scale'], 
                     holidays_prior_scale=params['holidays_prior_scale'],
                     yearly_seasonality = False,
                          weekly_seasonality = True,
                          daily_seasonality = False)
        model.fit(train)
        print('He salido')
        p_pred_train_y = model.predict(train) 
        p_pred_test_y = model.predict(test)
        
        y_tr = np.append(y_tr, train[['y']], axis=1)
        y_te = np.append(y_te, test[['y']], axis=1)
        pred_train_y = np.append(pred_train_y, p_pred_train_y[['yhat']], axis=1)
        pred_test_y = np.append(pred_test_y, p_pred_test_y[['yhat']], axis=1)
        
        # Performances of each shop
        if end_df <= date['date_end_train']:
            print(f'RMSLE_train {i}: ', np.round(np.sqrt(mean_squared_error(train[['y']].clip(0.0), p_pred_train_y[['ds', 'yhat']].set_index('ds').clip(0.0))), 4), f'RMSLE_test {i}: ', np.round(np.sqrt(mean_squared_error(test[['y']].clip(0.0), p_pred_test_y[['ds', 'yhat']].set_index('ds').clip(0.0))), 4))        
  

    # Total performances
    # Train
    if end_df <= date['date_end_train']:
        print(f'RMSLE_train tot: ', np.round(np.sqrt(mean_squared_error(y_tr.clip(0.0), pred_train_y.clip(0.0))), 4), f'RMSLE_test tot: ', np.round(np.sqrt(mean_squared_error(y_te.clip(0.0), pred_test_y.clip(0.0))), 4))
   
    
    return pred_test_y, y_te, par
