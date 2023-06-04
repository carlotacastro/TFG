def add_header_keys(df, keys, header):
    map_ = {k: header + k for k in keys}
    return df.rename(columns=map_)


def split_dynamic_stat_data(df, keys=None, thresh=0.05):
    if keys is None:
        keys = [k for k in df.keys() if k not in ['Agressions', 'Community Area', 'ds']]
    dynamic = []
    unique_codes = np.unique(df['Community Area'])
    limit = int(thresh * len(unique_codes))
    counts = {k: [] for k in keys}
    for code in unique_codes:
        code_df = df[df['Community Area'] == code]
        for k in keys:
            if len(np.unique(code_df[k])) > 1:
                #print(
                    #f'For key {k}, code {code} has voted that its dynamical. Unique values are {np.unique(code_df[k])}')
                counts[k].append(code)
                if len(counts[k]) >= limit:
                    #print(f'For key {k}, We have found its dynamical after codes {counts[k]} voted that its dynamical')
                    dynamic.append(k)
                    keys.remove(k)
    static = [k for k in keys]

    return dynamic, static



def split_real_categorical_data(df, thresh=0.1, keys=[], isdynamical=True, auto=False):
    """
    To decide if a variable is categorical or not, we can do the following.
    Take the length of all the values in a series and compare it with the unique values of the series.
    If the unique values is less than 10% of the length of the full series, it means that the variable is categorical.
    Else, the variable is dynamic.
    This nemotecnic rule should work pretty good.
    """

    if len(keys) == 0 and auto:
        keys = [k for k in df.keys() if k not in ['Agressions', 'Community Area', 'ds']]
    elif len(keys) == 0:
        return [], []

    if isdynamical:
        """
        this will not work for statical data... but it is the most pure way to do it.
        """
        votes = {k: 0 for k in keys}
        for code in np.unique(df['Community Area']):
            code_df = df[df['Community Area'] == code]
            for k in keys:
                if len(code_df[k]) > 1:
                    if len(np.unique(code_df[k])) < thresh * len(code_df[k]) or len(np.unique(code_df[k])) == 1:
                        votes[k] -= 1
                    else:
                        votes[k] += 1.5

        categorical = []
        real = []
        for k in votes.keys():
            if votes[k] >= 0:
                real.append(k)
            else:
                categorical.append(k)

    else:
        categorical = []
        for k in keys:
            if len(np.unique(df[k])) < thresh * len(df):
                categorical.append(k)
                keys.remove(k)
        real = keys

    print(f'Real keys before formatting {real}')
    print(f'Categorical keys before formatting {categorical}')
    print()
    for k in real:
        if isinstance(np.array(df[k])[0], str):
            categorical.append(k)
            real.remove(k)
    for k in categorical:
        if isinstance(np.array(df[k])[0], float):
            real.append(k)
            categorical.remove(k)
    print(f'Real keys after formatting {real}')
    print(f'Categorical keys after formatting {categorical}')
    return categorical, real



def split_features(df, thresh=0.1):
    dyn, stat = split_dynamic_stat_data(df)
    dyn_cat, dyn_re = split_real_categorical_data(df, thresh=thresh, keys=dyn)
    stat_cat, stat_re = split_real_categorical_data(df, thresh=thresh, keys=stat)
    return dyn_cat, dyn_re, stat_cat, stat_re

def generate_labelled_exogenous_df(df, one_hot=True, thresh=0.1,ignore_keys = [['Community Area','Agressions']]):
    df = df.fillna(0)
    dyn_cat, dyn_re, stat_cat, stat_re = split_features(df, thresh=thresh)
    print('Dynamical categorical features:')
    for k in dyn_cat:
        print(k)
    print('\n\n')
    print('Dynamical real features:')
    for k in dyn_re:
        print(k)
    print('\n\n')
    print('Statical categorical features:')
    for k in stat_cat:
        print(k)
    print('\n\n')
    print('Statical real features:')
    for k in stat_re:
        print(k)
    print('\n\n')
    new_df = df.copy()
    new_df = add_header_keys(new_df, dyn_cat, 'dyn_cat_')
    new_df = add_header_keys(new_df, dyn_re, 'dyn_re_')
    new_df = add_header_keys(new_df, stat_cat, 'stat_cat_')
    new_df = add_header_keys(new_df, stat_re, 'stat_re_')

    ordinal_encoder = None

    if one_hot:
        cat_dyn_keys = [k for k in new_df.keys() if 'dyn_cat_' in k]
        for k in cat_dyn_keys:
            tmp = pd.get_dummies(new_df[k], prefix=k)
            for new_k in tmp.keys():
                new_k_re = new_k.replace('dyn_cat_', 'dyn_re_OneHot_')
                new_df[new_k_re] = tmp[new_k].copy()
        new_df = new_df.drop(cat_dyn_keys, axis=1)

        cat_sat_keys = [k for k in new_df.keys() if 'stat_cat_' in k]
        features = new_df[cat_sat_keys].to_numpy()
        ordinal_encoder = OrdinalEncoder().fit(features)
        encoded_features = ordinal_encoder.transform(features)
        for i, k in enumerate(cat_sat_keys):
            new_df[k] = encoded_features[:, i]

    return new_df, ordinal_encoder
    
    
    
    
    
    class DeepAR() : 
    def __init__(self,df) : 

        """ 
        init (self,df) : the constructor of the class

        """
        self.df = df
    
    def prepare_dataset(self,start_index,target_index,target,freq):
        dataset = ListDataset([{"start": start_index, 
                                            "target": self.df[target][start_index:target_index]}], freq = freq)
        return dataset
    
    def prepare_dataset_grouped(self,start_index,target_index,target,freq, dyn_re_list, stat_cat_list, stat_re_list, groupby_index = 'Community Area'):
        dataset_list = []
        counter = 0
        total = len(self.df[groupby_index].unique())
        for item_id,df_item in self.df.groupby(groupby_index,as_index = False):
            counter+=1
            if counter%200==0:
                print (f'{counter}/{total}')
            current_dataset = {"start": start_index, 
                                            "target": self.df[target][start_index:target_index]}

            dynre_list = []
            statcat_list = []
            statre_list = []

            for dyn_re in dyn_re_list:
                dynre_list.append(self.df[dyn_re][start_index:target_index])

            for stat_cat in stat_cat_list:
                statcat_list.append(self.df[stat_cat].iloc[0])

            for stat_re in stat_re_list:
                statre_list.append(self.df[stat_re].iloc[0])


#             current_dataset[FieldName.FEAT_STATIC_CAT] = statcat_list
            current_dataset[FieldName.FEAT_DYNAMIC_REAL] = dynre_list
            current_dataset[FieldName.FEAT_STATIC_REAL] = statre_list

            dataset_list.append(current_dataset)



        dataset = ListDataset(dataset_list, freq = freq)

        return dataset  


    def prepare_train (self,start_index,target_index,target, freq, dyn_re_list, stat_cat_list, stat_re_list) : 
        self.training_data = self.prepare_dataset_grouped(start_index,target_index,target,freq, dyn_re_list, stat_cat_list, stat_re_list)

    def prepare_test (self,start_index,target_index,target,freq, dyn_re_list, stat_cat_list, stat_re_list) : 

        self.test_data = self.prepare_dataset_grouped(start_index,target_index,target,freq, dyn_re_list, stat_cat_list, stat_re_list)

    def estimator (self,freq,context_length,prediction_length,num_layers,num_cells,cell_type,epochs,cardinality, use_external = True) : 

        """"
        In order to fix the architecture of the estimator .. num_layers , cell_type etc ...

        """

        self.estimator = DeepAREstimator(freq=freq, 
                                context_length=context_length,  
                                prediction_length=prediction_length,
                                num_layers=num_layers,    
                                num_cells=num_cells,
                                cell_type=cell_type,
                                #trainer=Trainer(epochs=epochs,ctx=gpu())) 
                                trainer=Trainer(epochs=epochs),
                                use_feat_dynamic_real=True*use_external,
                                use_feat_static_cat=False*use_external,
                                use_feat_static_real=True*use_external,
                                distr_output = NegativeBinomialOutput())
#                                 cardinality = cardinality) 

        self.prediction_length = prediction_length

    def train(self) : 

        """
        to do the fit

        """
        self.predictor = self.estimator.train(training_data=self.training_data)

    def evaluate (self,num_samples=100) : 

        """ 
        to do evaluation task

        """

        forecast_it, ts_it = make_evaluation_predictions(
        dataset=self.test_data,  
        predictor=self.predictor,   
        num_samples=num_samples, 
        )
        self.forecasts = list(forecast_it)
        self.tss =  list(ts_it)


    def plot_prob_forecasts(self, index):

        """
        plot the forecasts

        """
        print('hello')
        prediction_intervals = (80.0, 95.0)
        legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        self.tss[index].plot(ax=ax) 
        self.forecasts[index].plot(color='g')
        #self.forecasts[index].plot(prediction_intervals=prediction_intervals, color='g')
        plt.grid(which="both")
        plt.legend(legend, loc="upper left")
        plt.show()
         
         
         
         
         
  def rmsle(y_pred,y):
    return np.sqrt(np.mean((np.log(y+1)-np.log(y_pred+1))**2))
    
    
    
    
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
