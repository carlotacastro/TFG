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
    df['firstday'] = df.day.apply(lambda x: 1 if x==1 else 0)
    df['primavera'] = df.month.apply(lambda x: 1 if x in (3,4,5) else 0)
    df['estiu'] = df.month.apply(lambda x: 1 if x in (6,7,8) else 0)
    df['tardor'] = df.month.apply(lambda x: 1 if x in (9,10,11) else 0)
    df['hivern'] = df.month.apply(lambda x: 1 if x in (12,1,2) else 0)
    
    
#maybe add holiday season


    df['daysinmonth'] = df.index.days_in_month.astype('int')
        
    # Dummy features
    df = pd.get_dummies(df, columns=['year'], drop_first=True)
    df = pd.get_dummies(df, columns=['quarter'], drop_first=True)
    df = pd.get_dummies(df, columns=['dayofweek'], drop_first=True)
    

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
    
        
    return df
    
    
    
    
    def features_monthly (data):
    
    df = data.copy()
        
    # Time features
    df['year'] = df.index.year.astype('int')
    df['quarter'] = df.index.quarter.astype('int')
    df['month'] = df.index.month.astype('int')
    df['primavera'] = df.month.apply(lambda x: 1 if x in (3,4,5) else 0)
    df['estiu'] = df.month.apply(lambda x: 1 if x in (6,7,8) else 0)
    df['tardor'] = df.month.apply(lambda x: 1 if x in (9,10,11) else 0)
    df['hivern'] = df.month.apply(lambda x: 1 if x in (12,1,2) else 0)
    
    # Adding lagsnew_{l}'].fillna(0)
        
    
#maybe add holiday season


        
    # Dummy features
    df = pd.get_dummies(df, columns=['year'], drop_first=True)
    df = pd.get_dummies(df, columns=['quarter'], drop_first=True)
    

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
    
        
    return df
