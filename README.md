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

## oil.csv
Preu diari del petroli. Inclou valors durant els períodes de temps de dades de l'entremanet i de test. (L'Equador és un país que depèn del petroli i la seva salut econòmica és molt vulnerable als xocs dels preus del petroli.)

### holidays_events.csv
Festius i Esdeveniments, amb metadades


Observacions importants sobre els conjunts de dades:

1. Un festiu que es transfereix oficialment cau aquest dia natural, però el govern va traslladar a una altra data. Un dia transferit s'assembla més a un dia normal que a un dia festiu. Per exemple, la festa de la Independència de Guayaquil es va traslladar del 2012-10-09 al 2012-10-12, el que significa que es va celebrar el 2012-10-12. Els dies del tipus _Bridge_ són dies addicionals que s'afegeixen a un festiu (p. ex., per allargar el descans durant un cap de setmana llarg). Sovint es componen pel tipus de dia de treball, que és un dia que normalment no està programat per a la feina (per exemple, dissabte) que pretén recuperar el pont.
2. Els dies festius addicionals són dies afegits a un dia festiu del calendari normal, per exemple, com succeeix normalment al voltant de Nadal (convertint la Nit de Nadal en un dia festiu).
3. Els salaris del sector públic es paguen cada dues setmanes el dia 15 i l'últim dia del mes. Les vendes dels supermercats es podrien veure afectades per això.
4. Un terratrèmol de magnitud 7,8 va afectar l'Equador el 16 d'abril de 2016. La gent es va concentrar en els esforços de socors donant aigua i altres productes de primera necessitat que van afectar molt les vendes dels supermercats durant diverses setmanes després del terratrèmol.
5. *******afegir més


## Objectius del dataset
Crear diversos models que prediguin amb més precisió les vendes per unitats de milers d'articles venuts a diferents botigues Favorita. El millor model es presentarà a la competició Kaggle volent trobar-se en el top 5% dels models.
