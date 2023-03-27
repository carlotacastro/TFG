def store (data):
    
    df = data.copy()
    
    # Adding features to stores
    df['uniquestore'] = df.city.apply(lambda x: 0 if x in ['Quito', 'Guayaquil', 'Santo Domingo', 'Cuenca', 'Manta', 'Machala', 'Latacunga', 'Ambato'] else 1)
    df['newstore'] = df.store_nbr.apply(lambda x: 1 if x in [19, 20, 21, 28, 35, 41, 51, 52] else 0)
        
    # Merging stores, test_df and train_df
    df = pd.concat([train_df, test_df], axis=0).merge(df, on=['store_nbr'], how='left')
    df = df.rename(columns={'type' : 'store'}) 

    return df
    
    




def holiday (data):
    
    df = data.copy()
    
    # Non-transferred events
    df.loc[297, 'transferred'] = df.loc[297, 'transferred'] = False
    df = df.query("transferred!=True")

    # 'Good Friday' mistake correction
    df['date'] = df['date'].replace({'2013-04-29' : pd.to_datetime('2013-03-29')}) 
    
    # Removing duplicates
    df = df.drop(index=holidays[holidays[['date', 'locale_name']].duplicated()].index.values)

    # Adding event type
    df.loc[df.type=='Event', 'type'] = df.description.apply(lambda x: x[0:7])
     
    # Selecting only the national holidays
    nat_df = df.query("locale=='National'")
    df = final_df.merge(nat_df, left_on=['date'], right_on=['date'], how='left')

    # Work days
    df['wd'] = True
    df.loc[df.type == 'Bridge'  , 'wd'] = False
    df.loc[df.type == 'Transfer', 'wd'] = False
    df.loc[(df.type == 'Additional') & (df.transferred == False), 'wd'] = False
    df.loc[(df.type == 'Holiday') & (df.transferred == False), 'wd'] = False  
    
    #Adding Easter
    easter_dates = ['2017-04-16', '2016-03-27', '2015-04-05', '2014-04-20', '2013-03-31']    
    df.loc[df['date'].isin(easter_dates), 'wd'] = False

    df['isevent'] = False
    df.loc[df.type == 'Event'  , 'isevent'] = True
    df.loc[df['date'].isin(easter_dates), 'isevent'] = True

   
    # Adding New Year
    df['firstday'] = df.description.apply(lambda x: 1 if x=='Primer dia del ano' else 0)
    df = df.drop(columns=['type', 'locale', 'locale_name', 'description', 'transferred'])

    # Adding closure days
    df['isclosed'] = df.groupby(by=['date', 'store_nbr'])['sales'].transform(lambda x: 1 if x.sum()==0 else 0)    
    df.loc[(df.date.dt.year==2017) & (df.date.dt.month==8) & (df.date.dt.day>=16) , 'isclosed'] = df.isclosed.apply(lambda x: 0)    
    df.loc[df.date.isin(['2017-01-01']), 'isevent'] = df.isevent.apply(lambda x: 'n')
    
    return df



  

def oil (data):
    
    df = data.copy()
    
    # Adding missing values
    df = df.set_index('date').resample("D").mean().interpolate(limit_direction='backward').reset_index()
    
    # Adding lags
    n_lags = [1, 2, 3, 4, 5, 6, 7, 10, 14]
    for l in n_lags:
        df[f'lagoil_{l}_dcoilwtico'] = df['dcoilwtico'].shift(l)

    df['oil_week_avg'] = df['dcoilwtico'].rolling(7).mean()
    df['oil_biweek_avg'] = df['dcoilwtico'].rolling(14).mean()
    df['oil_1_month_avg'] = df['dcoilwtico'].rolling(30).mean()
    df['oil_2_month_avg'] = df['dcoilwtico'].rolling(60).mean()

    df.dropna(inplace = True)
    
    # Merging oil and final_df
    df = final_df.merge(df, on=['date'], how='left')
    
    return df


  

def transaction (data):
    
    df = data.copy()
    
    # Merging transactions and final_df
    df = final_df.merge(df, on=['date', 'store_nbr'], how='left')
    
    # Filling missing values
    df.loc[(df.transactions.isnull()) & (df.isclosed==1), 'transactions'] = df.transactions.apply(lambda x: 0)

    # Filling missing values
    #average number of transactions per day per store
    group_df = df.groupby(by=['store_nbr', 'date']).transactions.first().reset_index()
    group_df['avg_tra'] = group_df.transactions.rolling(15, min_periods=10).mean()
    group_df.drop(columns='transactions', inplace=True)
    df = df.merge(group_df, on=['date', 'store_nbr'], how='left')
    df.loc[(df.transactions.isnull()) & (df.isclosed==0), 'transactions'] = df.avg_tra
    df.drop(columns='avg_tra', inplace=True)
    
    df.loc[(df.date.dt.year==2017) & (df.date.dt.month==8) & (df.date.dt.day>=16) , 'transactions'] = df.transactions.apply(lambda x: None)    

    return df





def features (data):
    
    df = data.copy()
        
    # Time features
    df['year'] = df.index.year.astype('int')
    df['quarter'] = df.index.quarter.astype('int')
    df['month'] = df.index.month.astype('int')
    df['day'] = df.index.day.astype('int')
    df['dayofweek'] = df.index.day_of_week.astype('int')
    df['weekofyear'] = df.index.week.astype('int')
    df['isweekend'] = df.dayofweek.apply(lambda x: 1 if x in (5,6) else 0)
    df['school_season'] = df.month.apply(lambda x: 1 if x in (4,5,8,9) else 0)
    
    df['daysinmonth'] = df.index.days_in_month.astype('int')

    # Wages in the public sector are paid every two weeks on the 15th and on the last day of the month
    df['wageday'] = ((df.index.is_month_end) | (df.index.day == 15))

    earthquake_affected_dates = [d for d in pd.date_range('2016-04-16', periods=20, freq='D').tolist()]
    df['earthquake_Aprin2016_effect'] = df.index.isin(earthquake_affected_dates)
    
    df['onpromotion_week_avg'] = df['onpromotion'].rolling(7).mean()
    df['onpromotion_biweek_avg'] = df['onpromotion'].rolling(14).mean()
    df['onpromotion_1_month_avg'] = df['onpromotion'].rolling(30).mean()
    df['onpromotion_2_month_avg'] = df['onpromotion'].rolling(60).mean()
    
    df['onpromotion_week_avg'] = df['onpromotion_week_avg'].fillna(0)
    df['onpromotion_biweek_avg'] = df['onpromotion_biweek_avg'].fillna(0)
    df['onpromotion_1_month_avg'] = df['onpromotion_1_month_avg'].fillna(0)
    df['onpromotion_2_month_avg'] = df['onpromotion_2_month_avg'].fillna(0)
        
    # Adding lags
    for l in range(1,5):
        df[f'onpromotion_lag_{l}'] = df['onpromotion'].shift(l)
        df[f'onpromotion_lag_{l}'] = df[f'onpromotion_lag_{l}'].fillna(0)
        

    # Dummy features
    df = pd.get_dummies(df, columns=['year'], drop_first=True)
    df = pd.get_dummies(df, columns=['quarter'], drop_first=True)
    df = pd.get_dummies(df, columns=['dayofweek'], drop_first=True)
    df = pd.get_dummies(df, columns=['store'], drop_first=True)
    df = pd.get_dummies(df, columns=['isevent'], drop_first=True)
    

    # DeterministicProcess
    fourierA = CalendarFourier(freq='A', order=5)
    fourierM = CalendarFourier(freq='M', order=2)
    fourierW = CalendarFourier(freq='W', order=4)

    dp = DeterministicProcess(index=df.index,
                          order=1,
                          seasonal=False,
                          constant=False,
                          additional_terms=[fourierA, fourierM, fourierW],
                          drop=True)
    dp_df = dp.in_sample()
    df = pd.concat([df, dp_df], axis=1)
    
    # Outliers
    df['outliers'] = df.sales.apply(lambda x: 1 if x>30000 else 0)
    
    df.drop(columns=['daysinmonth', 'month', 'state','city'], inplace=True)
    
    return df



  
def split_func (data, X, y, end_date, test_size):
    
    # Splitting train and test
    idx_train, idx_test = train_test_split(data.index, test_size=test_size, shuffle=False)
    X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]
    
    return X_train, y_train, X_test, y_test


def create_sample_weights(X, target_date, weight=0.9):
    extra_weight_days = X.index.get_level_values('date') > target_date
    return np.array(list(map(lambda x: np.exp(-weight) if x == 0 else 1, extra_weight_days.astype('int'))))



def feature_selection (data, end_df, n):
    
    df = data.loc[:end_df,:].reset_index().set_index(['store_nbr', 'family', 'date']).sort_index()
    y = np.log1p(df.loc[:,'sales'].unstack(['store_nbr', 'family']))
    
    # Selecting features
    X = df.drop(columns = ['id', 'sales', 'transactions', 'store_B', 'store_C', 'store_D', 'store_E'])
    X = X.groupby(by='date').first()
        
    # Train
    if end_df <= date['date_end_train']:
        y_tr = np.empty((93,0))
        y_te = np.empty((n,0))
        pred_train_y = np.empty((93,0))
        pred_test_y = np.empty((n,0))
    # Test
    else:
        y_tr = np.empty((diff_test-n,0))
        y_te = np.empty((n,0))
        pred_train_y = np.empty((diff_test-n,0))
        pred_test_y = np.empty((n,0))
        
    select = []

    # A model for each shop
    for i in data.store_nbr.unique():
        y = df.loc[i,'sales'].unstack(['family'])
        X = df.loc[i, X.columns]
        X = X.groupby(by='date').first()

        # Splitting train and test and log transformation
        X_train, y_train, X_test, y_test = split_func(y, X, np.log1p(y), end_df, n)
                
        # Exponentially weighted cost function
        weights = create_sample_weights(X_train, '2017-07-01') 
        
        # Random Forest Regressor
        sel = SelectFromModel(RandomForestRegressor(n_estimators=1200, max_depth = 50, max_features = 'auto', bootstrap = True, min_samples_leaf=2, min_samples_split=2, random_state=0))
        sel.fit(X_train, y_train, sample_weight = weights)
        selected_feat=X_train.columns[(sel.get_support())]
        select.append(selected_feat)
        model= RandomForestRegressor(n_estimators=1200, max_depth = 100, max_features = 'auto', bootstrap = True, min_samples_leaf=2, min_samples_split=2, random_state=0)
        model.fit(X_train[selected_feat], y_train, sample_weight = weights)
        rf_pred_train_y = model.predict(X_train[selected_feat]) 
        rf_pred_test_y = model.predict(X_test[selected_feat])
        
        y_tr = np.append(y_tr, y_train, axis=1)
        y_te = np.append(y_te, y_test, axis=1)
        pred_train_y = np.append(pred_train_y, rf_pred_train_y, axis=1)
        pred_test_y = np.append(pred_test_y, rf_pred_test_y, axis=1)
        
        # Performances of each shop
        # Train
        if end_df <= date['date_end_train']:
            print(f'RMSLE_train {i}: ', np.round(np.sqrt(mean_squared_error(y_train.clip(0.0), rf_pred_train_y.clip(0.0))), 4), f'RMSLE_test {i}: ', np.round(np.sqrt(mean_squared_error(y_test.clip(0.0), rf_pred_test_y.clip(0.0))), 4))
        

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
    pred_train_y = pred_train_y.stack(['store_nbr', 'family'])
    pred_test_y = pred_test_y.stack(['store_nbr', 'family'])
  
    return pred_test_y, y_te, select
