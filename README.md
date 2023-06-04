# TFG
### Nom: Carlota Loreto Castro Sánchez
### DATASET: Store Sales - Time Series Forecasting & Chicago Crime
### URL: [kaggle] https://www.kaggle.com/competitions/store-sales-time-series-forecasting & https://www.kaggle.com/datasets/chicago/chicago-crime
## Resum
Es tracta d'un Treball de Fi de Grau on s'analitzen sèries temporals de diferents datasets mitjançant diversos models paramètrics, aplicant els coneixements obtinguts per a la detecció i prevenció d'agressions sexuals.

El primer datatset a analitzar total està format per 6 datasets en format .csv:
- holiday_events.csv
- stores.csv
- transactions.csv
- oil.csv
- train.csv
- test.csv


El segon dataset a utilitzar es compon per un únic dataset anomenat crime.csv


Per a la primera competició, es tracta d'un conjunt de dades de Corporación Favorita, una gran botiga de queviures amb seu a l'Equador. Les dades de training inclouen dates, informació sobre la botiga i el producte, si aquest article s'estava promocionant, així com els números de vendes. Les dates es troben entre el 01/01/2013 al 31/08/2017.

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


Amb l'objectiu d'aprofundir en la predicció d'agressions sexuals, s'analitzarà la base de dades de Kaggle denominada _Chicago Crimes_, la qual registra incidents delictius denunciats (excloent els homicidis amb dades per a cada víctima) ocorreguts a la ciutat de Chicago des del 2001 fins a la present, excepte els set dies més recents. S'utilitza la base de dades pública de Chicago. Aquestes dades s'extreuen del sistema CLEAR (Citizen Law Enforcement Analysis and Reporting) del Departament de Policia de Chicago, i la darrera data disponible és el 8 de febrer del 2023, que correspon al moment en què es va començar a processar la base de dades. Cal tenir en compte que aquestes dades inclouen informes no verificats enviats al Departament de Policia, i les classificacions preliminars dels delictes poden canviar posteriorment com a resultat d'una investigació addicional. A més, sempre hi ha la possibilitat d'errors mecànics o humans\cite{Chicago Crimes}. Aquestes dades inclouen la localització del crim i les coordenades, el codi de l'FBI, el número de la comunitat, el barri, el districte, si es va realitzar l'arrest, si el crim va ser domèstic i el tipus de crims entre d'altres. Per fer l'anàlisi de les agressions sexuals seleccionarem de la base de dades únicament els crims classificats com _Sex Offense_ i _Crime Sexual Assaul_t.

## Objectius del dataset
Crear diversos models que prediguin amb més precisió les vendes per unitats de milers d'articles venuts a diferents botigues Favorita. El millor model es presentarà a la competició Kaggle volent trobar-se en el top 5% dels models  desprès acplicar els coneixements obtinguts per a la predicció d'agressions secuals.


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


Per al dataset de Chicago, una característica rellevant de la base de dades és la tendència a acumular un gran nombre d'agressions el primer dia de cada mes. Aquesta acumulació es pot explicar pel fet que les dades provenen del sistema CLEAR, on s'agrupen totes les agressions conegudes dins d'un mes determinat, però sense una data específica. El mateix fenomen s'observa en relació a l'any, on les agressions es registren el primer dia de l'any, i pel que fa a les hores, on les agressions es registren a les 12:01 a. m. o a les 12:00 a. m. Aquestes suposicions es fonamenten en l'existència d'un nombre extraordinari d'agressions a aquestes hores, que no es produeixen amb la mateixa freqüència en altres dies del mes.\\

S'ha dut a terme un estudi exhaustiu de les dades que revelen una alta incidència d'agressions en dies que no són el primer del mes. En primer lloc, s'han analitzat totes les notícies corresponents als dies amb un nombre total d'agressions superior a 15, així com la notícia del dia anterior, amb l'objectiu de determinar si un esdeveniment concret afectava el nombre d'agressions del dia següent. Les notícies s'han consultat a l'arxiu històric del diari _Chicago Sun Times_, un dels diaris més rellevants de la ciutat de Chicago.

S'ha elaborat una llista de les 10 notícies més rellevants de cada dia, agrupant-les per temàtiques. En una primera instància, cap esdeveniment concret ha semblat ser la causa de les agressions, ja que en un període de 22 anys no s'ha observat un patró específic relacionat amb un ambient polític més convuls, esdeveniments nacionals o regionals. No obstant això, s'ha observat que els dies amb notícies més rellevants (aquelles que han rebut un major nombre de visites) coincideixen amb esdeveniments esportius, especialment relacionats amb els equips de beisbol de Chicago, com ara _The Chicago Cubs_ i _The Chicago White Sox_, tot i que també s'inclouen en menor proporció l'equip de futbol americà _The Chicago Bears_ i l'equip de bàsquet _The Chicago Bulls_. A continuació, s'ha realitzat una comparativa entre els resultats dels dies anteriors als partits d'aquests equips i els resultats dels dies amb un nombre elevat d'agressions, amb l'objectiu de determinar una possible correlació mitjançant una anàlisi exhaustiva de l'historial dels resultats d'aquests equips en els últims 22 anys.

A més, s'han elaborat diferents atributs temporals per a la predicció del model (com l'any, el mes, el dia de la setmana, el dia, etc.). Cal destacar els següents atributs: New Years, que indica amb un valor de 1 els dies que corresponen al primer dia de l'any i amb 0 la resta de dies; firstday, que identifica el primer dia de cada mes; i isweekend, que assigna el valor de 1 als dies dissabte i diumenge i 0 als altres dies. Altres atributs que han contribuït a l'optimització del model inclouen l'ús de retardades en les agressions entre 1 i 5 dies, així com atributs de mitjanes de les agressions setmanals, bisetmanals, mensuals i bimensuals. Finalment, s'ha adoptat un procés determinista per a assegurar que el model produeixi sempre la mateixa sortida en funció de les mateixes condicions inicials o l'estat inicial.

És evident que les agressions sexuals augmenten durant les estacions més càlides, particularment a l'estiu, destacant els mesos de juliol i agost, i disminueixen durant les estacions més fredes. A més, la majoria de les agressions tenen lloc durant els caps de setmana. Aquests resultats concorden amb l'estudi realitzat pel Ministeri de l'Interior sobre les agressions sexuals, on es constata una tendència similar, tenint en compte que les dades de Chicago acumulen, al gener, les denúncies en les quals no s'especifica el mes ni el dia en què van tenir lloc les agressions.

D'altra banda, mitjançant un estudi sobre les hores mitjanes de llum mensuals a la ciutat de Chicago durant tot l'any, s'ha constatat que la majoria de les agressions tenen lloc durant la franja nocturna, representant un 55\% del total (cal destacar que s'han exclòs les agressions que es produeixen entre les 12 a.m. i les 12:01 a.m. per a normalitzar les dades de Chicago). Aquesta observació és concorde amb l'estudi realitzat a Catalunya sobre la prevenció d'agressions facilitades per drogues, on s'evidencia un increment d'agressions en horaris nocturns, especialment en entorns d'oci nocturn i durant els caps de setmana. A més, els autors d'aquest estudi coincideixen en destacar el consum d'alcohol com a factor de risc facilitador d'aquestes situacions.

A nivell de distribució geogràfica el 43\% de les agressions tenen lloc a la zona sud de Chicago (South Side), seguit per 29\% a la zona oest, 22\% a la zona nord i només un 6\% a la zona est i que la majoria de les agressions tenen lloc a domicilis o apartaments (el 55.56\%).

En l'estudi realitzat per investigar els patrons de pic d'agressions, utilitzant les notícies del diari Chicago Sun Times, no es van trobar correlacions clares. Tant els esdeveniments destacats dels dies analitzats com els resultats dels equips de futbol americà, bàsquet i beisbol no van demostrar ser factors causals. Així doncs, aquestes variables extrínseques no semblen ser conclusives en aquest context. Per a obtenir una comprensió més profunda de la situació de la ciutat, seria necessari disposar de dades més detallades que, lamentablement, no van ser accessibles durant el període d'investigació, tant per raons de temps com de recursos.

D'altra banda, altres variables es van formular basant-se en diferents estudis, com l'auditoria del sistema VioGen i el mapa nacional de solucions per a posar fi a la violència contra les dones. Aquests estudis han revelat que els col·lectius més afectats per l'agressió sexual i la violència de gènere són les dones immigrants, les desocupades, les persones grans, les amb discapacitat, les amb persones dependents a càrrec i les amb un baix nivell de competències digitals. Aquests col·lectius es troben especialment vulnerables, ja que tenen un coneixement limitat sobre la igualtat de gènere i es troben en situacions de desprotecció. Per aquest motiu, s'ha accedit a les dades de Chicago Data i s'ha extret informació sobre el percentatge de persones amb estudis secundaris i superiors, el nivell de pobresa i el nombre de persones nascudes a l'estranger, ja que aquestes dades han estat les més detallades disponibles.

## Entrenament + Testing
El primer que es fa es triar els atributs que hem decidit al Feature Selection que funcionen de manera més òptima per a cadascun dels nostres models. Realitzar l'entrenament com el testing per a cada botiga del nostre dataset (54 botigues) ha resultat en una millora significativa de les prediccions. També cal destacar que en afegir una funció de cost de pesos exponencial que feia émfasi en l'últim més de ventes abans de la predicció, el nostre error es va reduir encara més.

La evaluació es fa amb l'RMSLE (Root Mean Squared Logarithmic Error). Donat que el RMSLE penalitza més predir valors més baixos que alts de les vendes en comptes de calcular directament la mitjana, primer es fa el logaritme de les ventes i posteriorment s'inverteixen els valors, es a dir, y = np.expm1(mean(np.log1p(df.sales))). Aquesta es la y que es passa com a submission per a que calculi l'RMSLE.

També s'ha considerat necessari fer un procès d'Hiperparametr Tuning per a cadascun dels models i triar els millors hiperparàmetres.

Els models que s'han fet servir han estats els següents per ordre de rendiment:
1. Weighted Model Model (Es creen unes prediccions finals a partir de 30% de les prediccions del XGBoostRegressor i el RandomForestRegressor)
2. RandomForestRegressor
3. Stacking (RandomForestRegressor + XGBoostRegressor + LinearRegression)
4. Prophet
5. DeepAR


Per a Chicago Crimes es faran servir els mateixos models fent. Amb l'objectiu d'obtenir dades encara més precises, s'ha elaborat també un anàlisi i una predicció de les dades diàries, tant per a la ciutat en conjunt com per a les comunitats específiques. S'avaluaran amb més d'una mètrica, essent aquestes MAE, SMAPE, RMSE i Coverage.

## Rendiment dels models
| Model | Hiperparametres | Error |  
| -- | -- | -- |
| Random Forest | n_estimators=1200, max_depth = 50, max_features = 'auto', bootstrap = True, min_samples_leaf=2, min_samples_split=2, random_state=0 | 0.40591 |
| XGBoost | n_estimators=500, learning_rate = 0.01, max_depth= 3, subsample = 0.5, colsample_bytree = 0.4, colsample_bylevel = 1, random_state=0 | 0.42432 |
| Weighted Model | RF: n_estimators=1200, max_depth = 50, max_features = 'auto', bootstrap = True, min_samples_leaf=2, min_samples_split=2, random_state=0, XG: n_estimators=500, learning_rate = 0.01, max_depth= 3, subsample = 0.5, colsample_bytree = 0.6, colsample_bylevel = 1, random_state=0 | 0.40109  |
| Stacking | Mateixos paràmetres que a RF i a XGBoost|  0.41220 |
| Prophet | holidays = event_holiday, changepoint_prior_scale = 0.05, holidays_prior_scale = 0.01, seasonality_prior_scale = 0.01, seasonality_mode = 'additive', yearly_seasonality = False, weekly_seasonality = True, daily_seasonality = False | 3.24719 |
| DeepAR |||


Per a Chicago Crimes, cada columna representa un període on _Q1_ és la predicció del 24 de gener de 2023 al 6 de febrer del 2023, _Q2_ és la predicció del 18 de desembre de 2022 al 31 de desembre del 2022, _Q3_ és la predicció dels sis últims mesos de 2022 per al total de la ciutat, _Q4_ és la predicció de tot l'any 2022 per al total de la ciutat, _Q5_ és la predicció dels sis últims mesos de 2022 per comunitat, _Q6_ és la predicció de tot l'any 2022 per comunitat.

| Model | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 |
| -- | -- | -- | -- | -- | -- | -- |
| DeepAr | 27.81\% | 58.25\% | 6.22\% | 19.98\% | 93.03\% | 88.88\% |
| Prophet | 103.05\% | 54.19\% | 7.38\% | 7.44\% | 90.4\% | 89.44\% |
| XGBoost | 39.95\% | 45.04\% | 3.82\% | 4.97\% | 80.88\% | 78.55\% |
| Random Forest | **25.01\% ** | 50.36\% | 6.72\% | 5.94\% | 82.79\% | 80.10\% | 
| Weighted Model | 28.50\% | **49.26\% **| **3.64\%** | **5.06\%** | **81.18\%** | **77.8\%** |



| Model | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 |
| -- | -- | -- | -- | -- | -- | -- |
| DeepAr | 1.88 | 1.57 | 12.70 | 44.2 | 2.13 | 1.98 |
| Prophet | 4.04 | 1.44 | 14.59 | 14.86 | 2.10 | 2.05 |
| Random Forest | 1.58 | 1.39 | 13.99 | 11.5 | 1.36 | 1.26 |
| XGBoost | 2.15 | **1.26** | 7.80 | **9.81** | 1.43 | 1.38 |
| Weighted Model | **1.85** | 1.37 | **7.27** | 9.96 | **1.37** | **1.26** |




| Model | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 |
| -- | -- | -- | -- | -- | -- | -- |
| Prophet | 0\% | 45.86\% | 33.33\% | 33.33\% | 57.14\% | 57.25\% |
\hline
Random Forest | 57.14\% | 50\% | 66.67\% | 58.33\% | 58.22\% | 55.30\% |
\hline
Weighted Model | 64.29\% | 50\% | 33.33\% | 58.33\% | 61.26\% | 59.41\% \\
\hline 
DeepAr | 50\% | 21.43\% | 50\% | 41.67\% | 59.96\% | 58.12\%\\
\hline
XGBoost | 14.29\% | 57.14\% | 33.33\% | 50\% | 66.23\% | 68.72\%

| Model | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 |
| -- | -- | -- | -- | -- | -- | -- |
DeepAr | 2.84 | 1.76 | 16.77 | 53.34 | 3.04 | 2.83\\
\hline 
Prophet | 4.8 | 1.64 | 17.71 | 17.99 | 3.05 | 2.95\\
\hline
Random Forest | 2.60 | 1.78 | 14.87 | 15.96 | 1.89 | 1.75\\ 
\hline
XGBoost | 3.24 | 1.77 | 8.81 | \textbf{12.89} | 1.92 | 1.82\\
\hline
Weighted Model | \textbf{2.72} | \textbf{1.77} | \textbf{8.21} | 13.15 | \textbf{1.87} | \textbf{1.71} \\
