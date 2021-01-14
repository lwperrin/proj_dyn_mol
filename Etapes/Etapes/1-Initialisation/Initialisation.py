# DYNAMIQUE MOLECULAIRE 2D: Initialisation
import numpy as np
import matplotlib.pyplot as plt
import itertools # positionner les atomes
#
###############################################################################
####### paramètres de simulation: nombre d'atomes, de pas
# nombre de pas
npas = 1000
# nombre d'atomes par arrête
nba = 8
# nombre total d'atomes
nbtot = nba**2 ;
###############################################################################
############ Paramètres du potentiel interatomique ############################
# parametres du potentiel de Lennard-Jones (unite SI, pour l'argon)
sigma = 3.4e-10 #metre
epsilon = 1.65e-21 # joules
m = 6.69e-26 # kilogrammes
# distance du minimum de potentiel
re = 2.0**(1.0/6.0)*sigma
print("distance interatomique: %s Angstrom"%(re*1e10))
#
# rayon de coupure
rcut = 2.5*re
###############################################################################
# frontière de boite
fronti = 2
###############################################################################
# periode d'oscillation
puls0 = np.sqrt((57.1464*epsilon/(sigma**2.0))/m)
freq0 = puls0/(2.0*3.14159)
peri0 =1/freq0
print("periode oscillation atomique: %s s"%(peri0))
print("periode oscillation atomique: %s ps"%(peri0*1e12))
###############################################################################
# pas de temps
dt = peri0/75
print("pas de temps: %s ps"%(dt*1e12))
#
###############################################################################
# paramètres décidant du "film"
pfilm = 10
klist = range(0,npas,pfilm)
Film = False
###############################################################################
#
###############################################################################
########################### Force #############################################
###############################################################################
# definition de la force
def F(r):
    val = 4.0*epsilon*(6*sigma**6/(r**7.0)-12.0*sigma**12.0/(r**13.0))
    return val
###############################################################################
########## Position initiale des particules ###################################
###############################################################################
# rayon intiale
rini = 1.1*re
#
# liste de positions
listplos = np.arange(0,nba)
# placement des atomes  
coor = np.array(list(itertools.product(listplos, listplos))) 
coor = coor*rini ;
# coor[i]= les trois coordonnées de la particule i
# Pour tracer:
coort=np.transpose(coor) # coort[0] tous les x, coor[1] tous les y, coor[2] tous les z
# positions initiales  en figure 
plt.figure(0)
plt.plot(coort[0],coort[1] ,"ro")
plt.title("Position initiale")
plt.ylim(-fronti*re,(nba-1)*re+fronti*re)
plt.xlim(-fronti*re,(nba-1)*re+fronti*re)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.show()     
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
########################################
### création des parametres initiaux ###
v = coor*0 ;
v2 = v # vitesse au demi pas
Ec = np.zeros(nbtot) # initialisation de l'énergie cinétique de chaque particule
liaison = np.zeros(nbtot) # initialisation de liaison chaque particule
pk = np.zeros(npas) # liste des pas de temps
Eck = np.zeros(npas) # énergie cinétique totale à chaque pas
liaisonk = np.zeros(npas) # nombre total d'atomes liés à chaque pas
#
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
        v[j] = v2[j]+Fjval/(2*m)*dt  
        if j==0 :
            v[j] = 0*v[j] # immobilise une particule (empeche les mouvements de corps solide)
        ### création de la boîte
        if coor[j,0]>(nba-1)*re+fronti*re or coor[j,0]<-fronti*re :
            v[j,0] = - v[j,0]
        if coor[j,1]>(nba-1)*re+fronti*re or coor[j,1]<-fronti*re :
            v[j,1] = - v[j,1]
        # fin boite
        if np.sum(Fjval**2) == 0 :
            Ec[j] = 0 # ne prend pas l'EC de la particule en compte si elle n'a plus de liaison
            liaison[j] = 0
        else:
            Ec[j] = 0.5*m*(v[j,0]**2+v[j,1]**2)
            liaison[j] = 1
    # Eck
    Eck[k] = np.sum(Ec)
    liaisonk[k] = np.sum(liaison)
    pk[k] = k
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
###############################################################################
###############################################################################
###############################################################################
# sauvegarde
np.save('/Users/yanngueguen/Documents/Boulot/DEM/MD2DSimple/Etapes/2-Stabi/coordonne.npy', coor)
np.save('/Users/yanngueguen/Documents/Boulot/DEM/MD2DSimple/Etapes/2-Stabi/vitesse.npy', v)
# paramètre
parameter = np.array([sigma,epsilon,m,re,rcut,dt])
np.save('/Users/yanngueguen/Documents/Boulot/DEM/MD2DSimple/Etapes/2-Stabi/parameter.npy', parameter)
