import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.stats import norm, gaussian_kde
import seaborn as sns

nper_healthy_aff = 10;
nper_unhealthy_aff = 2;
t = np.arange(0,nhours,1)
## Normaux vs normaux recomposés

nper_healthy_aff = 10;
nper_unhealthy_aff = 3;
t = np.arange(0,nhours,1)

plt.figure(figsize=(12, 6))

plt.subplot(2,1,1)
plt.title("Normaux decomposition {} sur {} niveaux avec {} niveaux de reconstruction".format(waveletname,level_decomposition,level_reconstitution))
for i in range(nper_healthy_aff):
    plt.plot(t+nhours*i,signals_healthy[i,:])
    #plt.axvline(x=i*24, linestyle='--', color='red', alpha=0.7)

size_reconstitution = len(reconstructed_signals_healthy[0])
t2 = np.arange(0,size_reconstitution,1)
plt.subplot(2,1,2)
for i in range(nper_healthy_aff):
    plt.plot(t2+size_reconstitution*i,reconstructed_signals_healthy[85+i])

plt.show()

##



plt.subplot(2,1,1)
plt.title("Wavelet {}, approximation level {}".format(waveletname,level_decomposition-level_reconstitution))
for i in range(i):
    plt.plot(t,signals_healthy[i,:],label='day {}'.format(i))
    #plt.axvline(x=i*24, linestyle='--', color='red', alpha=0.7)

size_reconstitution = len(reconstructed_signals_healthy[10+i])
t2 = np.linspace(0,nhours,size_reconstitution)
plt.subplot(2,1,2)
for i in range(nper_healthy_aff):
    plt.plot(t2,reconstructed_signals_healthy[158+i])

plt.show()
## avec moyenne glissante
def moyenne_glissante(signal, fenetre):
    return np.convolve(signal, np.ones(fenetre)/fenetre, mode='valid')


def moyenne_glissante_2D(signal, fenetre):
    return np.apply_along_axis(lambda x: moyenne_glissante(x, fenetre), axis=1, arr=signal)

signals_healthy_mg = moyenne_glissante_2D(signals_healthy,3)

plt.figure(figsize=(12, 6))

plt.subplot(3,1,1)
plt.title("Normaux decomposition {} sur {} niveaux avec {} niveaux de reconstruction avec moyenne glissante".format(waveletname,level_decomposition,level_reconstitution))
for i in range(nper_healthy_aff):
    plt.plot(t+nhours*i,signals_healthy[i,:])
    #plt.axvline(x=i*24, linestyle='--', color='red', alpha=0.7)

size_reconstitution = len(reconstructed_signals_healthy[0])
t2 = np.arange(0,size_reconstitution,1)
plt.subplot(3,1,2)
for i in range(nper_healthy_aff):
    plt.plot(t2+size_reconstitution*i,reconstructed_signals_healthy[i])

t3 = np.arange(t[1],t[-1],1)
plt.subplot(3,1,3)
for i in range(nper_healthy_aff):
    plt.plot(t3+nhours*i,signals_healthy_mg[i,:])


plt.show()


## Oestrus vs oestrus recomposés

plt.figure(figsize=(12, 6))
plt.subplot(2,1,1)
plt.title("Oestrus decomposition {} sur {} niveaux avec {} niveaux de reconstruction".format(waveletname,level_decomposition,level_reconstitution))
for i in range(nper_unhealthy_aff):
    plt.plot(t+nhours*i,signals_unhealthy[i,:])
    plt.axvline(x=i*nhours, linestyle='--', color='red', alpha=0.7)

size_reconstitution = len(reconstructed_signals_unhealthy[0])
t2 = np.arange(0,size_reconstitution,1)
plt.subplot(2,1,2)
for i in range(nper_unhealthy_aff):
    plt.plot(t2+size_reconstitution*i,reconstructed_signals_unhealthy[i])

plt.show()

## normaux & oestrus VS recomposés
plt.figure(figsize=(12, 6))

plt.subplot(2,2,1)
plt.title("Healthy pre-decomposition")
for i in range(nper_healthy_aff):
    plt.plot(t,signals_healthy[82+369+i,:],label="day {}".format(i))
    #plt.axvline(x=i*nhours, linestyle='--', color='red', alpha=0.7)
plt.legend()
size_reconstitution = len(reconstructed_signals_healthy[0])
t2 = np.arange(0,size_reconstitution,1)
plt.subplot(2,2,3)
plt.title("Healthy post-decomposition")
for i in range(nper_healthy_aff):
    plt.plot(t2,reconstructed_signals_healthy[82+369+i],label="day {}".format(i))
plt.subplot(2,2,2)
plt.title("Oestrus pre-decomposition")
for i in range(nper_unhealthy_aff):
    plt.plot(t,signals_unhealthy[i*10,:])
    plt.axvline(x=i*nhours, linestyle='--', color='red', alpha=0.7)

plt.subplot(2,2,4)
plt.title("Oestrus post-decomposition")
for i in range(nper_unhealthy_aff):
    plt.plot(t2,reconstructed_signals_unhealthy[i*10])

plt.show()

##
#Vache 44432 12 mars 2015 et 17 juin 2015 oestrus
plt.figure(figsize=(12, 6))

plt.subplot(2,1,1)
plt.title("10 consecutives days")
for i in range(nper_healthy_aff):
    plt.plot(t+nhours*i,signals[82+369+85+i,:],label="day {}".format(i))
    plt.axvline(x=i*nhours, linestyle='--', color='red', alpha=0.7)
plt.legend()
size_reconstitution = len(reconstructed_signals_healthy[0])
t2 = np.arange(0,size_reconstitution,1)
plt.subplot(2,1,2)
plt.title("post-decomposition")
for i in range(nper_healthy_aff):
    plt.plot(t2,reconstructed_signals[82+369+85+i],label="day {}".format(i))
plt.subplot(2,2,2)
plt.title("2 days of oestrus pre-decomposition")
for i in range(nper_unhealthy_aff):
    plt.plot(t+nhours*i,signals_unhealthy[2+i,:])
    plt.axvline(x=i*nhours, linestyle='--', color='red', alpha=0.7)

plt.subplot(2,2,4)
plt.title("2 days of oestrus post-decomposition")
for i in range(nper_unhealthy_aff):
    plt.plot(t2,reconstructed_signals_unhealthy[2+i,:])

##



for i in range(nper_healthy_aff):
    plt.subplot(2,2,1)
    plt.plot(t+nhours*i,signals[82+369+88+i,:],label="day {}".format(i))
    plt.axvline(x=i*nhours, linestyle='--', color='red', alpha=0.7)
    plt.subplot(2,2,2)
    plt.plot(t,signals[82+369+88+i,:],label="day {}".format(i))
size_reconstitution = len(reconstructed_signals_healthy[0])
t2 = np.arange(0,size_reconstitution,1)
plt.subplot(2,2,1)
plt.title("10 activities rhythm of consecutives days")
plt.xlabel("hours")
plt.ylabel("activity level")
plt.legend()
plt.subplot(2,2,2)
plt.title("Superposition of these activities")
plt.xlabel("hours")
plt.subplot(2,1,2)
plt.title("Activities post wavelet transform")
for i in range(nper_healthy_aff):
    plt.plot(t2,reconstructed_signals[82+369+88+i],label="day {}".format(i))
    plt.legend()

plt.tight_layout
plt.show()
## Cartes des distances
# Trouver les valeurs minimales et maximales des deux matrices

vmin = min(min([np.min(x[x>0]) for x in distances_healthy.values()]),min([np.min(x[x>0]) for x in distances_healthy.values()]))
vmax = max(max([np.max(x) for x in distances_healthy.values()]),max([np.max(x) for x in distances_healthy.values()]))
# Afficher la première matrice avec la couleur de chaleur
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.imshow(distances_healthy[cow], cmap='hot', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
plt.title('Euclidian distances healthy vs healthy')

# Afficher la deuxième matrice avec la couleur de chaleur
plt.subplot(1, 2, 2)
plt.imshow(distances_unhealthy[cow], cmap='hot', interpolation='nearest', aspect='auto',vmin=vmin, vmax=vmax)
plt.title('Euclidian distances healthy (row) vs oestrus (column)')

# Ajouter une barre de couleur commune
plt.colorbar()

mean_healthy = np.mean(distances_healthy[cow][distances_healthy[cow]>0])
mean_unhealthy = np.mean(distances_unhealthy[cow])
std_healthy = np.std(distances_healthy[cow][distances_healthy[cow]>0])
std_unhealthy = np.std(distances_unhealthy[cow])

# plt.text(-40,120,'Moyenne : {} \nVariance : {}'.format(mean_healthy,std_healthy))
# plt.text(0,0.2,'Moyenne : {} \nVariance : {}'.format(mean_unhealthy,std_unhealthy),ha='center', va='center', transform=plt.gca().transAxes)

print("Moyenne normal vs normal : ",mean_healthy)
print("Variance normal vs normal : ",std_healthy)
print("Moyenne normal vs oestrus : ",mean_unhealthy)
print("Variance normal vs oestrus : ",std_unhealthy)

# Afficher les deux matrices côte à côte

plt.show()

##
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.stats import norm, gaussian_kde
import seaborn as sns


cow = cows_oestrus[1]
# Générer des données de test (remplacez cela par vos données réelles)

donnees = distances_healthy[cow][distances_healthy[cow]>0]
donnees2 = distances_unhealthy[cow].ravel()
#donnees = np.concatenate([x[x>0].ravel() for x in distances_healthy.values()])
#donnees2 = np.concatenate([x.ravel() for x in distances_unhealthy.values()])

mean_healthy = np.mean(donnees)
mean_unhealthy = np.mean(donnees2)
std_healthy = np.std(donnees)
std_unhealthy = np.std(donnees2)

# plt.text(-40,120,'Moyenne : {} \nVariance : {}'.format(mean_healthy,std_healthy))
# plt.text(0,0.2,'Moyenne : {} \nVariance : {}'.format(mean_unhealthy,std_unhealthy),ha='center', va='center', transform=plt.gca().transAxes)

print("Moyenne normal vs normal : ",mean_healthy)
print("Variance normal vs normal : ",std_healthy)
print("Moyenne normal vs oestrus : ",mean_unhealthy)
print("Variance normal vs oestrus : ",std_unhealthy)


kde1 = gaussian_kde(donnees)
kde2 = gaussian_kde(donnees2)



# pred1 = (donnees>seuil).astype(int)
# pred2 = (donnees2>seuil).astype(int)
# pred = np.concatenate([pred1,pred2])
# vraies_classes = np.concatenate([np.zeros_like(pred1),np.ones_like(pred2)])
# matrice_confusion = confusion_matrix(vraies_classes, pred)
#
# sns.heatmap(matrice_confusion, annot=True, fmt='d', cmap='Blues', cbar=False,
#             xticklabels=['Classe A', 'Classe B'], yticklabels=['Classe A', 'Classe B'])
# plt.xlabel('Prédictions')
# plt.ylabel('Vraies classes')
# plt.title('Matrice de Confusion')
# plt.show()




# Ajuster une distribution normale aux données
plt.xlim([0,10000])
plt.ylim([0,0.001])
x = np.linspace(0, 10000, 1000)
kde1x = kde1(x)
kde2x = kde2(x)

intersection = np.minimum(kde1x, kde2x)
# Calculer l'aire sous les courbes positives
area1 = np.trapz(kde1x, x)
area2 = np.trapz(kde2x, x)

# Calculer l'aire de l'intersection
intersection_area = np.trapz(intersection, x)

# Calculer le pourcentage d'intersection
intersection_percentage1 = (intersection_area / area1) * 100
intersection_percentage2 = (intersection_area / area2) * 100
# Tracer l'histogramme des moyennes
plt.hist(donnees, bins=30, density=True, alpha=0.6, color='b')
plt.hist(donnees2, bins=30, density=True, alpha=0.6, color='r')


plt.plot(x,kde1x,linewidth=2,label="Healthy VS Healthy")

plt.plot(x,kde2x,linewidth=2,label="Healthy VS Oestrus")

plt.legend()


# Ajouter des annotations et des étiquettes
title = "Cow {}\nApproximation with {} level {}\n".format(cow,waveletname,level_decomposition-level_reconstitution) + "Healthy: mean = %.2f, std = %.2f\nOestrus: mean = %.2f, std = %.2f" % (mean_healthy, std_healthy,mean_unhealthy, std_unhealthy)
plt.title(title)
plt.xlabel('distances')
plt.ylabel('Density')
plt.show()
##
P = np.zeros((ncows_oestrus,2))
for i in range(ncows_oestrus):

    cow = cows_oestrus[i]
    # Générer des données de test (remplacez cela par vos données réelles)

    donnees = distances_healthy[cow][distances_healthy[cow]>0]
    donnees2 = distances_unhealthy[cow].ravel()
    #donnees = np.concatenate([x[x>0].ravel() for x in distances_healthy.values()])
    #donnees2 = np.concatenate([x.ravel() for x in distances_unhealthy.values()])

    mean_healthy = np.mean(donnees)
    mean_unhealthy = np.mean(donnees2)
    std_healthy = np.std(donnees)
    std_unhealthy = np.std(donnees2)


    kde1 = gaussian_kde(donnees)
    kde2 = gaussian_kde(donnees2)

    x = np.linspace(0, 10000, 1000)
    kde1x = kde1(x)
    kde2x = kde2(x)

    intersection = np.minimum(kde1x, kde2x)
    # Calculer l'aire sous les courbes positives
    area1 = np.trapz(kde1x, x)
    area2 = np.trapz(kde2x, x)

    # Calculer l'aire de l'intersection
    intersection_area = np.trapz(intersection, x)

    # Calculer le pourcentage d'intersection
    intersection_percentage1 = (intersection_area / area1) * 100
    intersection_percentage2 = (intersection_area / area2) * 100

    P[i,:] = [intersection_percentage1,intersection_percentage2]

plt.figure(figsize=(14, 6))
plt.imshow(np.reshape(P[:,1],(19,7)), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()
