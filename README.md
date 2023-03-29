# TFG
### Nom: Carlota Loreto Castro Sánchez
### DATASET: Store Sales - Time Series Forecasting
### URL: [kaggle] https://www.kaggle.com/competitions/store-sales-time-series-forecasting
## Resum
Es tracta d'un Treball de Fi de Grau on s'analitzen sèries temporals de diferents datasets mitjançant diversos models paramètrics.

El primer datatset a analitzar total està format per 6 datasets en format .csv:
- holiday_events.csv
- stores.csv
- transactions.csv
- oil.csv
- train.csv
- test.csv


Es tracta d'un conjunt de dades de Corporación Favorita, una gran botiga de queviures amb seu a l'Equador. Les dades de training inclouen dates, informació sobre la botiga i el producte, si aquest article s'estava promocionant, així com els números de vendes. Les dates es troben entre el 01/01/2013 al 31/08/2017.

### train.csv
Les dades d'entrenament, que inclouen sèries temporals de funcions store_nbr, family i onpromotion, així com les vendes objectiu.
- _store_nbr_ identifica la botiga on es venen els productes.
- _family_ identifica el tipus de producte venut.
- _sales_ dóna el total de vendes d'una família de productes en una botiga concreta en una data determinada. Els valors fraccionats són possibles ja que els productes es poden vendre en unitats fraccionades (1,5 kg de formatge, per exemple, en lloc d'1 bossa de patates fregides).
- _onpromotion_ proporciona el nombre total d'articles d'una família de productes que s'estaven promocionant en una botiga en una data determinada.


### test.csv
Les dades de test, amb les mateixes característiques que les dades d'entrenament. Servex per a predir les vendes objectiu per a les dates.
Les dates de les dades de la prova són per als 15 dies posteriors a l'última data de les dades d'entrenament (del 16/08/2017 al 31/08/2017).


### stores.csv
Emmagatzema metadades, com ara ciutat, estat, tipus i clúster( una agrupació de botigues similars).

### oil.csv
Preu diari del petroli. Inclou valors durant els períodes de temps de dades de l'entremanet i de test. (L'Equador és un país que depèn del petroli i la seva salut econòmica és molt vulnerable als xocs dels preus del petroli.)

### holidays_events.csv
Festius i Esdeveniments, amb metadades


Observacions importants sobre els conjunts de dades:

1. Un festiu que es transfereix oficialment cau aquest dia natural, però el govern va traslladar a una altra data. Un dia transferit s'assembla més a un dia normal que a un dia festiu. Per exemple, la festa de la Independència de Guayaquil es va traslladar del 2012-10-09 al 2012-10-12, el que significa que es va celebrar el 2012-10-12. Els dies del tipus _Bridge_ són dies addicionals que s'afegeixen a un festiu (p. ex., per allargar el descans durant un cap de setmana llarg). Sovint es componen pel tipus de dia de treball, que és un dia que normalment no està programat per a la feina (per exemple, dissabte) que pretén recuperar el pont.
2. Els dies festius addicionals són dies afegits a un dia festiu del calendari normal, per exemple, com succeeix normalment al voltant de Nadal (convertint la Nit de Nadal en un dia festiu).
3. Els salaris del sector públic es paguen cada dues setmanes el dia 15 i l'últim dia del mes. Les vendes dels supermercats es podrien veure afectades per això.
4. Un terratrèmol de magnitud 7,8 va afectar l'Equador el 16 d'abril de 2016. La gent es va concentrar en els esforços de socors donant aigua i altres productes de primera necessitat que van afectar molt les vendes dels supermercats durant diverses setmanes després del terratrèmol.
5. Triem que les dades comencin el 30-4-2017 ja que la última botiga obre el 24-4-2017. Donem un temps de marge per a estabilitzar les ventes
6. S'ha aplicat una transformació logarítmica a les vendes per solucionar un problema de linealitat mostrat en la trama residual, probablement a causa del creixement compost.
7. L'entrenament i predicció es fa calculant l'error per a cada botiga tenint en compte les families de ventes.


## Objectius del dataset
Crear diversos models que prediguin amb més precisió les vendes per unitats de milers d'articles venuts a diferents botigues Favorita. El millor model es presentarà a la competició Kaggle volent trobar-se en el top 5% dels models.


## Preprocessat
El primer que he realitzat ha estat la neteja de les dades de cadascun dels datasets per separat donat que es tracta d'un dataset molt extens. Desprès de cada neteja s'ajunta cada dataset al final. Per a portar aquesta acció a terme he realitzat diferent mètodes:

1. Es crea la categoria _unique store_ per a identificar aquelles botigues que són exclusives d'una localitat. També es crea la categoria _new store_ per a les botigues que han obert passada la data d'inici del dataset
2. Seleccionem unicament aquells esdeveniments que no han estat transferits i fem correccions en esdeveniments amb dates mal assignades per tal de tenir una concordança en les dades.
3. Eliminem els duplicats
4. Només agafem les festes com a tipus nacionals. No es categoritzen entre locals o regionals ja que sembla impactar de forma negativa el rendiment.
5. Es genera l'atribut _wd_ _(work day)_ per a definir els dies laborals (True) dels festius (False), així com l'atribut _isclosed_
6. S'afageix dates importants com Pasqua o el primer dia de l'any com atributs.
7. S'afeigeixen valors nulls al dataset de l'oli i es creen diversos lags per fer que els seus valors passats semblin contemporanis amb els valors que estem intentant predir (fa que les sèries retardades siguin útils com a característiques per modelar la dependència en sèrie) així com atributs que indiquen la mitjana de l'oli semanal, bisemanal, mensual,etc.
8. S'omplen els valors buits de les transactions amb el nombre mitjà de transaccions per dia per botiga
9. Finalment, s'afageixen noves característiques de temps, incloent els períodes en què s'inicien les escoles (abril-maig i agost-setembre. Important per a captar l'estacionalitat de les vendes de productes escolars), els dies on es paguen als treballadors, _wageday_, el 15 i el 30 de cada mes i les dates del terratremol d'abril del 2016.
10. Una altra característica que ha tingut un efecte molt positiu en la predicció de les dades és incloure atributs sobre els productes en promoció, _onpromotion_. S'han inclòs atributs que indiquen la mitjana dels prodcutes en promoció semanals, bisemanals, mensuals, etc. així com les mitjanes de productes en promoció per a cada botiga i lags.
11. També es fa servir un procès determinista per a produir sempre la mateixa sortida a partir de les mateixes condicions de partida o l'estat inicial.

## Entrenament + Testing
El primer que es fa es triar els atributs que hem decidit al Feature Selection que funcionen de manera més òptima per a cadascun dels nostres models. Realitzar l'entrenament com el testing per a cada botiga del nostre dataset (54 botigues) ha resultat en una millora significativa de les prediccions. També cal destacar que en afegir una funció de cost de pesos exponencial que feia émfasi en l'últim més de ventes abans de la predicció, el nostre error es va reduir encara més.

La evaluació es fa amb l'RMSLE (Root Mean Squared Logarithmic Error). Donat que el RMSLE penalitza més predir valors més baixos que alts de les vendes en comptes de calcular directament la mitjana, primer es fa el logaritme de les ventes i posteriorment s'inverteixen els valors, es a dir, y = np.expm1(mean(np.log1p(df.sales))). Aquesta es la y que es passa com a submission per a que calculi l'RMSLE.

També s'ha considerat necessari fer un procès d'Hiperparametr Tuning per a cadascun dels models i triar els millors hiperparàmetres.

Els models que s'han fet servir han estats els següents per ordre de rendiment:
1. Average Model (Es creen unes prediccions finals a partir de 30% de les prediccions del XGBoostRegressor i el RandomForestRegressor)
2. RandomForestRegressor
3. Stacking (RandomForestRegressor + XGBoostRegressor + LinearRegression)
4. Prophet
5. DeepAR


## Rendiment dels models
| Model | Hiperparametres | Error |  
| -- | -- | -- |
| Random Forest | n_estimators=1200, max_depth = 50, max_features = 'auto', bootstrap = True, min_samples_leaf=2, min_samples_split=2, random_state=0 | 0.40591 |
| XGBoost | n_estimators=500, learning_rate = 0.01, max_depth= 3, subsample = 0.5, colsample_bytree = 0.4, colsample_bylevel = 1, random_state=0 | 0.42432 |
| Average Model | RF: n_estimators=1200, max_depth = 50, max_features = 'auto', bootstrap = True, min_samples_leaf=2, min_samples_split=2, random_state=0, XG: n_estimators=500, learning_rate = 0.01, max_depth= 3, subsample = 0.5, colsample_bytree = 0.6, colsample_bylevel = 1, random_state=0 | 0.40109  |
| Stacking | Mateixos paràmetres que a RF i a XGBoost|  0.41220 |
| Prophet |||
| DeepAR |||

## Conclusions

## Idees per a treballar en un futur

