import numpy as np
import pywt
import random as rd
import matplotlib.pyplot as plt
from fastdtw import fastdtw


def get_consecutive_dates(serie_dates):
    # to get series of consecutives dates
    if serie_dates.empty:
        return []

    serie_dates_triees = serie_dates.sort_values()
    ensembles_dates = []
    ensemble_dates_actuel = [serie_dates_triees.iloc[0]]

    for i in range(1, len(serie_dates_triees)):
        date_actuelle = pd.to_datetime(serie_dates_triees.iloc[i])
        date_precedente = pd.to_datetime(serie_dates_triees.iloc[i - 1])

        if (date_actuelle - date_precedente).days == 1:
            ensemble_dates_actuel.append(serie_dates_triees.iloc[i])
        else:
            ensembles_dates.append(ensemble_dates_actuel)
            ensemble_dates_actuel = [serie_dates_triees.iloc[i]]

    ensembles_dates.append(ensemble_dates_actuel)

    return ensembles_dates


def to_ts_24h(serie):
    #reshape a serie to get time series of 24 hours
    return np.reshape(serie,(len(serie)//24,24))

ncows = len(cows);
AR = dict()
ARhealthy = dict()
ARoestrus = dict()

k=0
for i in range(ncows):
    ARhealthy[cows[i]]=[]
    datacow = datacows[cows[i]]
    datacow_healthy = datacow[datacow["OK"]==1]
    cons_dates = get_consecutive_dates(datacow_healthy["date"].drop_duplicates())
    for serie in cons_dates:
        ts = to_ts_24h(datacow_healthy[datacow_healthy["date"].isin(serie)]["ACTIVITY_LEVEL"])
        ARhealthy[cows[i]].append(ts)


    ARoestrus[cows[i]] = to_ts_24h(datacow[datacow["oestrus"]==1]["ACTIVITY_LEVEL"])
    AR[cows[i]]=to_ts_24h(datacow["ACTIVITY_LEVEL"])

## Parametres

nper_healthy= 0
for x in ARhealthy.values():
    for d in x:
        nper_healthy+=np.size(d,0)
nper_unhealthy = 0
for x in ARoestrus.values():
    nper_unhealthy+=np.size(x,0)
nhours = 24;

#nper_healthy = 100;
#nper_unhealthy = 10;
waveletnames = ['haar','db2','db3','bior1.3','bior2.2','bior3.1','rbio2.2','rbio3.1','coif1']
waveletname = waveletnames[1]
dec_len = pywt.Wavelet(waveletname).dec_len
level_decomposition = pywt.dwt_max_level(nhours,waveletname);
print("level_decomposition : ",level_decomposition)
level_reconstitution = 0;

signals_healthy = np.zeros((nper_healthy,nhours))

# kcow,ks,kj=0,0,0
# for i in range(nper_healthy):
#     done = False
#     while not(done):
#         cow = cows[kcow]
#         serie = AR[cow][ks].ravel()
#         if len(serie[kj:kj+nhours]) < nhours:
#             ks+=1
#             kj=0;
#             if ks == len(AR[cow]):
#                 kcow+=1
#                 ks=0
#                 kj=0
#         else:
#             signals_healthy[i,:] = serie[kj:kj+nhours]
#             kj+=nhours
#             done = True
#
# signals_unhealthy = np.zeros((nper_unhealthy,nhours))
#
# kcow,ks,kj=0,0,0
# for i in range(41):
#     done = False
#     while not(done):
#         cow = cows[kcow]
#         serie = ARoestrus[cow][ks].ravel()
#         if len(serie[kj:kj+nhours]) < nhours:
#             ks+=1
#             kj=0;
#             if ks >= len(ARoestrus[cow]):
#                 kcow+=1
#                 ks=0
#                 kj=0
#         else:
#             signals_unhealthy[i,:] = serie[kj:kj+nhours]
#             kj+=nhours
#             done = True

#signals_healthy = np.reshape(AR.ravel()[:(nhours*nper_healthy)],(nper_healthy,nhours))
#signals_unhealthy = np.reshape(ARoestrus.ravel()[:(nhours*nper_unhealthy)],(nper_unhealthy,nhours))

signals_healthy = np.concatenate([np.concatenate(x) for x in ARhealthy.values()])
signals_unhealthy = np.concatenate([x for x in ARoestrus.values()])
signals = np.concatenate([x for x in AR.values()])

##

coeffs = pywt.wavedec(signals_healthy[0,:], waveletname, level=level_decomposition)
coeffs = coeffs[:level_reconstitution+1]
len_reconst = np.size(pywt.waverec(coeffs, waveletname),0)

reconstructed_signals_healthy = np.zeros((nper_healthy,len_reconst))
reconstructed_signals_unhealthy = np.zeros((nper_unhealthy,len_reconst))
reconstructed_signals = np.zeros((np.size(signals,0),len_reconst))

for i in range(nper_healthy):
    # Decomposition
    coeffs = pywt.wavedec(signals_healthy[i,:], waveletname, level=level_decomposition)

    # Keep approximation coeff until a certain level
    coeffs = coeffs[:level_reconstitution+1]

    # Recomposition
    reconstructed_signals_healthy[i] = pywt.waverec(coeffs, waveletname)


for i in range(nper_unhealthy):
    # Decomposition
    coeffs = pywt.wavedec(signals_unhealthy[i,:], waveletname, level=level_decomposition)

    # Keep approximation coeff until a certain level
    coeffs = coeffs[:level_reconstitution+1]

    # Recomposition
    reconstructed_signals_unhealthy[i] = pywt.waverec(coeffs, waveletname)

for i in range(np.size(signals,0)):
    # Decomposition
    coeffs = pywt.wavedec(signals[i,:], waveletname, level=level_decomposition)

    # Keep approximation coeff until a certain level
    coeffs = coeffs[:level_reconstitution+1]

    # Recomposition
    reconstructed_signals[i] = pywt.waverec(coeffs, waveletname)


## Toutes les vaches les unes des autres

# distances_healthy = np.zeros((nper_healthy,nper_healthy))
# distances_unhealthy = np.zeros((nper_healthy,nper_unhealthy))
# for i in range(nper_healthy):
#     for j in range(i,nper_healthy):
#         distances_healthy[i,j] = np.linalg.norm(reconstructed_signals_healthy[i]-reconstructed_signals_healthy[j])
#         #distances_healthy[i,j] = fastdtw(reconstructed_signals_healthy[i],reconstructed_signals_healthy[j])[0]
# for i in range(nper_healthy):
#     for j in range(nper_unhealthy):
#         distances_unhealthy[i,j] = np.linalg.norm(reconstructed_signals_healthy[i]-reconstructed_signals_unhealthy[j])
#         #distances_unhealthy[i,j] = fastdtw(reconstructed_signals_healthy[i],reconstructed_signals_unhealthy[j])[0]



## Effet vache : on compare la vache i avec ses courbes et sa courbe d'oestrus

distances_healthy = dict()
distances_unhealthy = dict()

k1,k2=0,0
for icow in range(ncows):
    cow = cows[icow]

    n1 = np.size(np.concatenate(ARhealthy[cow]),0)
    n2 = np.size(ARoestrus[cow],0)
    distances_healthy[cow] = np.zeros((n1,n1))
    distances_unhealthy[cow] = np.zeros((n1,n2))
    for i in range(n1):
        for j in range(i,n1):
            distances_healthy[cow][i,j] = np.linalg.norm(reconstructed_signals_healthy[k1+i]-reconstructed_signals_healthy[k1+j])
    k1+=n1
    for i in range(n1):
        for j in range(n2):
            distances_unhealthy[cow][i,j] = np.linalg.norm(reconstructed_signals_healthy[k2+i]-reconstructed_signals_unhealthy[k2+j])
    k2+=n2


#decomposition pour que ça se ressemble pour les jours
# trouver la marge
#decomposition sur des paquets de jours
# regarder
# jour en commun
# chercher la taille de la fenêtre

##
#1 seul vache
#trouver le bon nombre de jours pour trouver une ressemblance