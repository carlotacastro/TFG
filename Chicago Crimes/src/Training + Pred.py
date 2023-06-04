def df_column_uniquify(df):
    df_columns = df.columns
    new_columns = []
    for item in df_columns:
        counter = 0
        newitem = item
        while newitem in new_columns:
            counter += 1
            newitem = "{}_{}".format(item, counter)
        new_columns.append(newitem)
    df.columns = new_columns
    return df
    
    
    
    def split_func (data, X, y, end_date, test_size):
    
    # Splitting train and test
    idx_train, idx_test = train_test_split(data.index, test_size=test_size, shuffle=False)
    X_train, y_train = X.loc[idx_train, :], X.loc[idx_test, :]
    X_test, y_test = y.loc[idx_train], y.loc[idx_test]
    
    return X_train, y_train, X_test, y_test
    
    
    
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
                          yearly_seasonality = True,
                          weekly_seasonality = True,
                          daily_seasonality = False)

        print('He entrado', p)
        train_model.fit(train)
        p_pred_test_y = train_model.predict(test)
        MAE = np.mean(np.abs(mean_squared_error(f.iloc[-14:].to_frame(), p_pred_test_y[['ds', 'yhat']].set_index('ds').clip(0.0))))
        model_parameters = model_parameters.append({'MAE':MAE,'Parameters':p},ignore_index=True)

    parameters = model_parameters.sort_values(by=['MAE'])
    parameters = parameters.reset_index(drop=True)
    
    return parameters['Parameters'][0]
    
    
    def sqr_err(y_true, y_pred):
    """

    :param y_true: true values of y
    :param y_pred: predicted values of y
    :return: array of lenght original data containing mean squared error for each predictions
    """
    if len(y_true) != len(y_pred):
        raise IndexError("Mismathced array sizes, you inputted arrays with sizes {} and {}".format(len(y_true),
                                                                                                  len(y_pred)))
    else:
        length = len(y_true)

    sqrerror_out = [(y_pred[i]-y_true[i])**2 for i in range(length)]

    return np.array(sqrerror_out)
    
    
    def split_func_com (data, X, y, end_date, test_size):
    
    # Splitting train and test
    idx_train, idx_test = train_test_split(data.index, test_size=test_size, shuffle=False)
    X_train, y_train = X.loc[idx_train.unique(), :], X.loc[idx_test.unique(), :]
    X_test, y_test = y.loc[idx_train.unique()], y.loc[idx_test.unique()]
    
    return X_train, y_train, X_test, y_test
