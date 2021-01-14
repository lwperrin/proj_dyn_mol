# DYNAMIQUE MOLECULAIRE 2D: Stabilisation vers 3K
import numpy as np
import matplotlib.pyplot as plt
#
# Chargement des paramètres sauvegardés
coor = np.load('coordonne.npy')
v = np.load('vitesse.npy')
parameter = np.load('parameter.npy')
# nombre d'atomes
nbtot = int(np.size(coor)/2)
print("Nombre atomes: %s "%(nbtot))
nba = nbtot**0.5
#
#################################
## paramètres de Lannard-Jones ##
#################################
sigma = parameter[0]
epsilon = parameter[1]
m = parameter[2]
re = parameter[3]
rcut = parameter[4]
dt = parameter[5]
###############################################################################
########################### Force #############################################
###############################################################################
# definition de la force
def F(r):
    val = 4.0*epsilon*(6*sigma**6/(r**7.0)-12.0*sigma**12.0/(r**13.0))
    return val
###############################################################################
###############################################################################
###############################################################################
################### Fonction de calcul des foces entre particules #############
###############################################################################
#### calcul des distances et forces pour une particule j
def Fj(j) :
    Fxi = 0
    Fyi = 0
    disti = coor-coor[j] # contient les distance x et y de la particule j aux autres
    xi = np.transpose(disti) # distance selon x
    xi = xi[0]
    yi = np.transpose(disti) # distance selon y
    yi = yi[1]
    disti2 = disti**2 # mets ces distances au carré
    ri2 = np.sum(disti2,1) # contient les distances ri au carré de la particle j aux autres
    ri = ri2**0.5 # et les ri
    ri[ri==0] = rcut # remplace les distances nulles par rcut
    ri[ri>rcut] = rcut #  remplace les distances >rcut par rcut
    Fi = F(ri) # toutes les forces subie par j des particules i
    Fi[Fi==F(rcut)] = 0 # remplace les forces pour r>rcut par 0
    Fxi = xi/ri*Fi
    Fyi = yi/ri*Fi
    Fx = np.sum(Fxi)
    Fy = np.sum(Fyi)
    return np.array([Fx,Fy])
###############################################################################
###############################################################################
##############################
# frontière de boite
fronti = 2
##############################
## nombre de pas
npas = 5000
##############################
# positions initiales en figure 
coort = np.transpose(coor)
plt.figure(0)
plt.plot(coort[0],coort[1] ,"ro")
plt.title("Position initiale")
plt.ylim(-fronti*re,(nba-1)*re+fronti*re)
plt.xlim(-fronti*re,(nba-1)*re+fronti*re)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.show()   
###############################
###############################################################################
# paramètres décidant du "film"
pfilm = 50
klist = range(0,npas,pfilm)
Film = False
###############################################################################
########################################
### création des parametres initiaux ###
v2 = v # vitesse au demi pas
Ec = np.zeros(nbtot) # initialisation de l'énergie cinétique de chaque particule liées
EcT = np.zeros(nbtot) # initialisation de l'énergie cinétique de chaque particule 
liaison = np.zeros(nbtot) # initialisation de liaison chaque particule
pk = np.zeros(npas) # liste des pas de temps
Eck = np.zeros(npas) # énergie cinétique totale à chaque pas des particules liée
EckT = np.zeros(npas) # énergie cinétique totale à chaque pas des particules 
liaisonk = np.zeros(npas) # nombre total d'atomes liés à chaque pas
Tk = np.zeros(npas) # liste de température de l'ensemble
Tks = np.zeros(npas) # liste de température de la partie solide
#
###############################################################################
#############################     Température      ############################
###############################################################################
gamma = 0.5 # parametre pour asservir la temperature ("potard")
betaC = True # True si la temperature est controlee, False sinon
#
# constante de Boltzmann pour les calculs de temperatures
kB=1.38064852e-23 #unite SI
# rampe de température voulue
Tini = 40 # température initiale
Tfini = 3 # température finale
def Tvoulue(k):
    return (Tfini-Tini)*k/npas+Tini
T = Tini # initilisation température
beta = 0 # initialisation beta
###############################################################################
###############################################################################
###############################################################################
#
###########################################
########## giga-boulces massives ##########
###########################################
for k in range(npas):
    vmean = np.transpose(np.mean(np.transpose(v),1))
    v = v-vmean # empecher un mouvement du centre de gravité
    for j in range (nbtot) :
        v2[j] = v[j]+Fj(j)/(2*m)*dt
        if j==int(nbtot/2) :
            v2[j] = 0*v2[j] # immobilise une particule (empeche les mouvements de corps solide)
        coor[j] = coor[j]+v2[j]*dt
        Fjval = Fj(j) # stocke la valeur de Fj(j) pour ne pas la recalculer
        v[j] = (v2[j]+Fjval/(2*m)*dt)*(beta+1) # le beta controle la température  
        ### création de la boîte
        if coor[j,0]>(nba-1)*re+fronti*re or coor[j,0]<-fronti*re :
            v[j,0] = - v[j,0]
        if coor[j,1]>(nba-1)*re+fronti*re or coor[j,1]<-fronti*re :
            v[j,1] = - v[j,1]
        # fin boite
        EcT[j] = 0.5*m*(v[j,0]**2+v[j,1]**2)
        if np.sum(Fjval**2) == 0 :
            Ec[j] = 0 # ne prend pas l'EC de la particule en compte si elle n'a plus de liaison
            liaison[j] = 0
        else:
            Ec[j] = EcT[j]
            liaison[j] = 1
    # Eck
    Eck[k] = np.sum(Ec)
    EckT[k] = np.sum(EcT)
    liaisonk[k] = np.sum(liaison)
    pk[k] = k
    ##########################################################################
    # calcul et asservissement temperature
    T = 2.0*EckT[k]/(kB*2.0*nbtot) # température de l'ensemble
    Ts = 2.0*Eck[k]/(kB*2.0*liaisonk[k]) # température des atomes liés
    Tk[k] = T
    Tks[k] = Ts
    print("temperature: %s K"%(T))
    print("temperature solide: %s K"%(Ts))
    beta=np.sqrt(1+gamma*(Tvoulue(k)/T-1))-1
    beta = beta*betaC # pour controler la temperature
    ###########################################################################
    # affichages
    print("Pas: %s"%(k))
    # dessin
    if Film : # booléen vrai ou faux
        if k in klist:
            coort=np.transpose(coor)
            plt.ioff() # pour ne pas afficher les graphs)
            fig = plt.figure()
            plt.plot(coort[0],coort[1] ,"ro")
            plt.title("Position initiale")
            plt.ylim(-fronti*re,(nba-1)*re+fronti*re)
            plt.xlim(-fronti*re,(nba-1)*re+fronti*re)
            plt.xlabel("x", fontsize=14)
            plt.ylabel("y", fontsize=14)
            fig.savefig('StorePic/MD-picture%s.png' % k) # sauvegarde incrementale
            plt.close(fig) # fermeture du graph
###############################################################################
###############################################################################
###############################################################################
#################### Figures ##################################################
coort=np.transpose(coor)
plt.figure(1)
plt.plot(coort[0],coort[1] ,"ro")
plt.title("Position Finale")
plt.ylim(-fronti*re,(nba-1)*re+fronti*re)
plt.xlim(-fronti*re,(nba-1)*re+fronti*re)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.show()
#
plt.figure(2)
plt.plot(pk,Eck ,"b")
plt.title("Energie")
plt.xlabel("Pas", fontsize=14)
plt.ylabel("Energie cinétique", fontsize=14)
plt.show()
#
plt.figure(3)
plt.plot(pk,liaisonk ,"b")
plt.title("Liaison")
plt.xlabel("Pas", fontsize=14)
plt.ylabel("Nombre atomes en liaison", fontsize=14)
plt.show()
#
plt.figure(4)
plt.plot(np.arange(nbtot),Ec ,"bo")
plt.title("Distribution")
plt.xlabel("Particule", fontsize=14)
plt.ylabel("Ec", fontsize=14)
plt.show()
# dessin des températures
plt.figure(5)
line_T, = plt.plot(pk,Tk,'b')
line_Ts, = plt.plot(pk,Tks,'r')
plt.xlabel('pas')
plt.ylabel('Temperature (K)')
plt.legend([line_T, line_Ts], ['T', 'T solide'])
plt.show()
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
# sauvegarde
np.save('/Users/yanngueguen/Documents/Boulot/DEM/MD2DSimple/Etapes/3-Fonte/coordonne.npy', coor)
np.save('/Users/yanngueguen/Documents/Boulot/DEM/MD2DSimple/Etapes/3-Fonte/vitesse.npy', v)
# paramètre
parameter = np.array([sigma,epsilon,m,re,rcut,dt])
np.save('/Users/yanngueguen/Documents/Boulot/DEM/MD2DSimple/Etapes/3-Fonte/parameter.npy', parameter)