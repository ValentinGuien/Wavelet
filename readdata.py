import numpy as np
import pandas as pd
from datetime import datetime,timedelta

filename = '../Data/dataset4-1.csv'

# Configurate the ACTIVITY_LEVEL so we look at from 0:00 + offset to 23:00 + offset
offset = 4

data = pd.read_csv(filename) # Read csv
data = data.drop(['IN_ALLEYS','REST','EAT','acidosis'],axis=1) # Remove acidosis
data['hour'] = data['hour'].astype(str)
#data['time'] = pd.to_datetime(data['date'] + '-' + data['hour']) # datetime of day + hour

# Keep cows with a minimal amount of 14 days of data
occ = data['cow'].value_counts()
data = data[data['cow'].isin(occ[occ>=24*14].index)]

cows = data["cow"].unique() # List of cows
states = ['oestrus','calving','lameness','mastitis','other_disease', 'accidents','disturbance','mixing','management_changes','OK'] # all the states
if filename != '../Data/dataset1-1.csv':
    data = data.drop(['LPS'],axis=1)
else:
    states.append('LPS')

def get_consecutive_dates(tableau_dates):
    if len(tableau_dates) == 0:
        return []

    tableau_dates_triees = np.sort(tableau_dates)
    ensembles_dates = []
    ensemble_dates_actuel = [tableau_dates_triees[0]]

    for i in range(1, len(tableau_dates_triees)):
        date_actuelle = np.datetime64(tableau_dates_triees[i])
        date_precedente = np.datetime64(tableau_dates_triees[i - 1])

        if (date_actuelle - date_precedente).astype(int) == 1:
            ensemble_dates_actuel.append(tableau_dates_triees[i])
        else:
            ensembles_dates.append(ensemble_dates_actuel)
            ensemble_dates_actuel = [tableau_dates_triees[i]]

    ensembles_dates.append(ensemble_dates_actuel)

    return ensembles_dates

#Shift
for cow in cows:
    datacow = data[data['cow']==cow]
    cons_dates = get_consecutive_dates(datacow['date'].unique())
    for serie in cons_dates:
        datacowserie = datacow[datacow['date'].isin(serie)]
        data.loc[datacowserie.index[0]:datacowserie.index[-1],"ACTIVITY_LEVEL"] = data.loc[datacowserie.index[0]:datacowserie.index[-1],"ACTIVITY_LEVEL"].shift(-offset)

data = data.dropna()

## Rules for fuzzy data

# 'state:[-p,q]' means that the p days before and q days after the labelled day of the event
# are considered as fuzzy days

fuzzy_rules = {'oestrus':[-1,2],
               'calving':[-2,7],
               'lameness':[-2,7],
               'mastitis':[-2,7],
               'LPS':[0,7],
               'other_disease':[-2,7],
               'accidents':[0,7],
               'disturbance':[0,0],
               'mixing':[0,7],
               'management_changes':[0,0]}

## Non sane cows

abn_data = data[data["OK"]!=1] # Dataframe containing only the cows and days where the state is not OK
cows_oestrus = abn_data[abn_data["oestrus"]==1]['cow'].unique()
ncows_oestrus = len(cows_oestrus)
## Fuzzy data

def maxinterval(I1,I2):
    return [min(I1[0],I2[0]),max(I1[1],I2[1])]

def getfuzzyrule(row):
    # Obtain the fuzzy rules associated for a row (a row can have more than one anomaly)
    fuz = [0,0]
    for s in states[:-1]:
        if row[s]==1:
            fuz = maxinterval(fuz,fuzzy_rules[s])
    return fuz

def getfuzzydays(row):
    # Obtain the fuzzy days associated for a row
    fuzzyrule = getfuzzyrule(row)
    date = pd.to_datetime(row["date"])
    return [date + timedelta(days=i) for i in range(fuzzyrule[0],fuzzyrule[1]+1) if i!=0]

def getfuzzydata():

    cowdateabn = abn_data.drop_duplicates(['cow','date']) # List of cow/day of anomaly
    nabn = cowdateabn.shape[0]

    #Creation of dataframe that will contain the fuzzy data
    n20 = 20*nabn # pre-allocation of 20 times the number of cow/day of anomaly
    fuzzy_data = pd.DataFrame({'cow':[0 for i in range(n20)],'date':['' for i in range(n20)]})


    # Completion of the dataframe fuzzy_data
    k = 0
    for i in range(nabn):
        row = cowdateabn.iloc[i,:]
        fuzzydays = getfuzzydays(row)
        m = len(fuzzydays)
        for j in range(m):

            values = [row['cow'],fuzzydays[j].strftime('%Y-%m-%d')]
            couple_existe = cowdateabn[(cowdateabn['cow'] == values[0]) & (cowdateabn['date'] == values[1])].shape[0] > 0
            if not(couple_existe):
                fuzzy_data.iloc[k,:] = values
                k+=1

    fuzzy_data = fuzzy_data.drop(index=range(k, len(fuzzy_data)))

    return fuzzy_data.drop_duplicates()

fuzzy_data = getfuzzydata();



## Healthy cows
def exclude_df(df,df_to_not_check):
    # Use function merge with how='outer'
    merged_df = pd.merge(df, df_to_not_check, how='outer', on=['cow', 'date'], indicator=True)

    # Filter rows where there isn't correspondance between DataFrames
    result_df = merged_df[merged_df['_merge'] == 'left_only']

    # Suppress "_merge"
    return result_df.drop('_merge', axis=1)


datacows = dict();
for cow in cows:
     datacow = data[data['cow']==cow]
     dates = datacow['date'].unique()
     for d in dates:
         if sum(datacow['date']==d) < 24:
             datacow = datacow[datacow['date']!=d]
     datacow = exclude_df(datacow,fuzzy_data)
     datacows[cow]=datacow

##
# AR = datacow["ACTIVITY_LEVEL"]
# AR = np.reshape(AR,(len(AR)//24,24))
#
# ARoestrus = abn_data[abn_data["oestrus"]==1]["ACTIVITY_LEVEL"]
# ARoestrus = np.reshape(ARoestrus,(len(ARoestrus)//24,24))