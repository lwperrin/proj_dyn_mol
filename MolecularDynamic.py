from scipy import * # pour utiliser la fonction exponentielle ici
import matplotlib.pyplot as plt # pour les graph
from math import * # les maths...
import numpy as np
import numexpr as ne # ne.evaluate("") si j ai compris favorise le multicoeur
import time # calcul du... temps de calcul
#
# Ce programme utilise:
# le potentiel de Lennard-Jones
# la methode de Velocity-Verlet pour cacluler les positions
# la methode de velicity rescaling pour asservir la temperature
#
# parametres du potentiel de Lennard-Jones (unite SI, pour l'argon)
sigma = 3.4e-10 #metre
epsilon = 1.65e-21 # joules
m = 6.69e-26 # kilogrammes
# position du minimum de potentiel
re = 2.0**(1.0/6.0)*sigma
print("distance interatomique: %s Angstrom"%(re*1e10))
#
# rayon de coupure (limite standard = 2*re)
rcut = 2.0*re
#
# deca: ecart initial entre atome en fracion de re
deca = 0.95
#
# periode d'oscillation pour pouvoir calibrer le pas de temps
puls0 = np.sqrt((57.1464*epsilon/(sigma**2.0))/m)
freq0 = puls0/(2.0*3.14159)
peri0 =1/freq0
print("periode oscillation atomique: %s s"%(peri0))
print("periode oscillation atomique: %s ps"%(peri0*1e12))
#
# pas de temps: 
dt = peri0/75
print("pas de temps: %s ps"%(dt*1e12))
#
# nombre d'atomes sur x
nbrex = 15
# nombre d'atomes sur y
nbrey = 15
# nombre d'atomes au total
npart = nbrex*nbrey
# nombre de pas
npas = 2000
#
# limite droite de boite (position)
XlimD = (nbrex-1)*re*deca+0.5*re*deca
# limite gauche boite (position)
XlimG = -0.5*re*deca
# limite haute de boite (position)
YlimH = (nbrey-1)*re*deca+0.5*re*deca
# limite basse boite (position)
YlimB = -0.5*re*deca
# dimension de la boite de calcul
LengthX = XlimD-XlimG
LengthY = YlimH-YlimB
Volume = LengthX*LengthY*re # volume avec une epaisseur re
#
# pour le film, afficher une image simule sur:
pfilm = 20
# enregistrer les images du film?
film = True
#
# calcul de la masse simulee
masseSim = m * npart
print("masse simulee: %s kg"%(masseSim))
# masse volumique
densi = masseSim/Volume
print("masse volumique: %s kg/m3"%(densi))
#
# temperature voulue, on peut programmer ce qu'on veut: ici un cosinus
DeltaT = 100 # Kelvin amplitude
perioT = 1.0*npas # periode en pas de temps
gamma = 0.5 # parametre pour asservir la temperature ("potard")
betaC = True # True si la temperature est controlee, False sinon
#
# creation des positions initiales: maille carre de cote re
# positions selon x: on cree la grille reguliere
posx = np.concatenate([np.arange(0,nbrex,1.0) for i in range(nbrey)])*re*deca
posx0 = posx # pour le calcul MSD on garde la position initiale en memoire
# positions selon y: on cree la grille reguliere
posy = np.concatenate([i*np.ones(nbrex) for i in range(nbrey)])*re*deca
posy0 = posy # pour le calcul MSD
#
# longueur x et y pour la periodicite
shiftX = LengthX-rcut # un atome a droite d'un autre, plus loin que ca, sera considere comme a gauche et vice-versa
shiftY = LengthY-rcut # meme principe verticalement
#
### calcul pour initier le premier pas (vitesses aleatoire)
# il n'est pas indispensable d avoir une vitesse non nulle, mais ca ajoute de la dynamique
# temperature initiale visee
Tini = 1.0 # kelvin
# vitesse distribuee en loi normale (moyenne 0.0, ecart type 0.001)
vx =np.random.normal(0.0, 0.001, npart)
vy =np.random.normal(0.0, 0.001, npart)
meanV2 = np.mean(ne.evaluate("vx*vx+vy*vy"))
# constante de Boltzmann pour les calculs de temperatures
kB=1.38064852e-23 #unite SI
vfact = np.sqrt(2*kB*Tini/(m*meanV2)) # pourconvertir la vitesse (l'energie cinetique) en temperature voulue
vx =vx*vfact
vy =vy*vfact
# estimation de la temperature correspondante pour initier la rampe
EC=ne.evaluate("0.5*m*(vx*vx+vy*vy)")
EC=ne.evaluate("sum(EC)")
Tvoulue = ne.evaluate("2.0*EC/(kB*2.0*npart)")
print("temperature initiale: %s K"%(Tvoulue))
#
# trace des positions et vecteurs deplacement initiaux (deduit de la vitesse initiale)
plt.figure(1)
plt.quiver([np.mean(posx)],[np.mean(posy)],[np.mean(vx*dt)],[np.mean(vy*dt)],color='g',angles='xy', scale_units='xy', scale=1) # le deplacement global de toutes les particules
plt.quiver(posx,posy,vx*dt*200,vy*dt*200,(vx*vx+vy*vy),angles='xy', scale_units='xy', scale=1) # le deplacement de chaque particule
plt.plot(posx, posy,'ro',markersize=5)
plt.ylim(YlimB,YlimH)
plt.xlim(XlimG,XlimD)
plt.show(block=False) # true empeche l'excecution de la suite du programme avant fermeture de la fenetre
plt.close
#
# les forces au premier pas sont nulles et de la dimension de posx (posy)
Fx = ne.evaluate("0.0*posx")
Fy = ne.evaluate("0.0*posy")
#
# enregistrement des positions dans des fichiers,
# c pas propre mais ca fonctionne: a modifier
fichx = open('storex.npy', 'a+b')
np.save(fichx, posx)
fichy = open('storey.npy', 'a+b')
np.save(fichy, posy)
#
# initialisation listes fonctions du pas
pask = [] # le pas lui-meme
pasCPU = [] # le temps de calcul par pas
pasEC = [] # energie cinetique
pasEP = [] # energie potentielle
pasET = [] # energie totale
pasT = [] # temperature
pasTC = [] # temperature de consigne
pastemps = [] # le temps
pasLiai = [] # nbre liaisons par atome
pasMSD = [] # MSD
tempstot = 0 #initialisation du chrono
#
for k in range(npas):
    pask += [k] # stockage des pas
    pastemps +=[k*dt*1e12] # definition du pas de temps en ps
    debutk = time.perf_counter() # calcul le temps de calcul
    print('%%%%%%%%%%%%%%')
    print("pas numero: %s"%(k))
    #
    # la rampe de T
    Tvoulue = 0.5*(DeltaT*cos(2*math.pi/perioT*k-math.pi)+DeltaT)+1+Tini
    pasTC += [Tvoulue]
    #
    # methode de velocity-verlet
    # vitesse au demi pas de temps
    vx2 = ne.evaluate("vx+(Fx/(2*m)*dt)")
    vy2 = ne.evaluate("vy+(Fy/(2*m)*dt)")
    # nouvelles positions deduites
    posx=ne.evaluate("posx+(vx2*dt)")
    posy=ne.evaluate("posy+(vy2*dt)")
    #
    # Mean Square Displacement
    MSDX = ne.evaluate("posx-posx0")
    MSDY = ne.evaluate("posy-posy0")
    MSD = ne.evaluate("MSDX**2.0+MSDY**2.0")
    MSD = ne.evaluate("sum(MSD)")
    MSD = MSD/npart
    MSD = MSD/(re**2.0)
    pasMSD += [MSD]
    #
    # correction de systeme periodic: on replace les atomes trop a droite a gauche, etc...
    # correction droite
    maskLimD = ne.evaluate("posx>XlimD")
    corrXD = -LengthX
    corrXD = ne.evaluate("maskLimD*corrXD")
    posx = ne.evaluate("posx+corrXD")
    # correction gauche
    maskLimG = ne.evaluate("posx<XlimG")
    corrXG = LengthX
    corrXG = ne.evaluate("maskLimG*corrXG")
    posx = ne.evaluate("posx+corrXG")
    # correction haute
    maskLimH = ne.evaluate("posy>YlimH")
    corrYH = -LengthY
    corrYH = ne.evaluate("maskLimH*corrYH")
    posy = ne.evaluate("posy+corrYH")
    # correction basse
    maskLimB = ne.evaluate("posy<YlimB")
    corrYB = LengthY
    corrYB = ne.evaluate("maskLimB*corrYB")
    posy = ne.evaluate("posy+corrYB")
    #
    # stockage des positions dans les fichiers deja cree
    np.save(fichx, posx)
    np.save(fichy, posy)
    #
    # creation matrice de position relative entre particule i et j selon X:
    # RX(i,j)=X(i)-X(j), matrice anti-symetrique a diagonale nulle
    # il faut creer un matrice MX dont chaque colonne est X(i) et alors:
    # RX=MX-transpose(MX)
    # creationd de MX
    MX = [posx for i in range(npart)]
    # calcul de RX
    MXT = np.transpose(MX)
    RX=ne.evaluate("MX-MXT")
    del MXT # efface pour liberer memoire
    #
    # forces periodiques
    # ajoute une force a gauche due aux particules de droite
    maskRD = ne.evaluate("RX>shiftX") # si atomes j plus loin a droite de l'atome i que shiftX
    maskRD = ne.evaluate("maskRD*LengthX")
    RX = ne.evaluate("RX-maskRD") # on le considere comme a gauche en retirant la longueur de cellule de sa position
    del maskRD
    # ajout une force a droite: meme principe
    maskRG = ne.evaluate("RX<(-shiftX)")
    maskRG = ne.evaluate("maskRG*LengthX")
    RX = ne.evaluate("RX+maskRG")
    del maskRG
    #
    # meme principe en Y
    # creation de MY
    MY = [posy for i in range(npart)]
    # calcul de RX
    MYT=np.transpose(MY)
    RY=ne.evaluate("MY-MYT")
    del MYT # efface pour liberer memoire
    # force periodique
    # ajoute une force en bas due aux particules en haut
    maskRH = ne.evaluate("RY>shiftY")
    maskRH = ne.evaluate("maskRH*LengthY")
    RY = ne.evaluate("RY-maskRH")
    del maskRH
    # ajout une force en haut
    maskRB = ne.evaluate("RY<(-shiftY)")
    maskRB = ne.evaluate("maskRB*LengthY")
    RY = ne.evaluate("RY+maskRB")
    del maskRB
    #
    # calcul de R, distance entre particule
    R=ne.evaluate("sqrt(RX*RX+RY*RY)")
    #
    # la diagonale de R est nulle, ca ne nous arrange pas, ca cree des
    # forces infinies. On remplace la diagonale par plus que le rcut
    R=R+(np.eye(npart)*2.0*rcut)
    #
    # creation d un masque pour virer tout ce qui depasse du rcut
    mask = ne.evaluate("(R < rcut)")
    #
    # calcul de la matrice des forces F (symetrique):
    # F(i,j)=force de la particule j sur i (scalaire)
    F =ne.evaluate("-4.0*epsilon*(6*sigma**6/(R**7.0)-12.0*sigma**12.0/(R**13.0))")
    # gain de ne.evaluate: pour 100x100 particules 5 s: 20% sur mon mac 8Go RAM 2 coeur
    # taille de F
    print("taille de F: %s bit"%(F.nbytes))
    #
    # on retire les forces au dela de rcut, dont celle de la diagonale truquee du coup
    F = ne.evaluate("F*mask")
    #
    # projection sur x et y
    RX=ne.evaluate("RX*mask") # on remasque par pure precaution...
    RY=ne.evaluate("RY*mask") # on remasque par pure precaution...
    Fx = ne.evaluate("F*RX/R")
    Fy = ne.evaluate("F*RY/R")
    #
    # resultante par atome (somme de force j subit par l'atome i)
    Fx = ne.evaluate("sum(Fx,axis=0)")
    Fy = ne.evaluate("sum(Fy,axis=0)")
    #
    # calcul energie cinetique
    EC=ne.evaluate("sum(0.5*m*(vx*vx+vy*vy))")
    print("Energie cinetique : %s J"%(EC)) # affichage
    #
    # caclul energie potentielle
    EP =ne.evaluate("epsilon*(4.0*(sigma**12.0/(R**12.0)-sigma**6.0/(R**6.0))+127/4096)")
    EP = mask*EP*0.5 # le 0.5 vient du fait des doubles interactions i-j j-i
    EP =ne.evaluate("sum(EP)")
    print("Energie potentielle : %s J"%(EP)) # affichage
    #
    # calcul energie totale
    ET = EC+EP
    #
    # calcul liaison par atom
    pasLiai += [sum(mask)/(2.0*npart)]
    #
    # stockage des energies
    pasEC += [EC]
    pasEP += [EP]
    pasET += [ET]
    print("Energie totale : %s J"%(EP)) # affichage
    #
    # calcul et asservissement temperature
    T = 2.0*EC/(kB*2.0*npart)
    print("temperature: %s K"%(T))
    beta=np.sqrt(1+gamma*(Tvoulue/T-1))-1
    beta = beta*betaC # pour controler ou non la temperature
    # stockage de la temperature
    pasT += [T]
    #
    # calcul du nouveau vx ou vy (methode de Verlet-vitesse) pour le pas suivant
    vx = ne.evaluate("(vx2+(Fx/(2*m)*dt))*(beta+1)") # le beta fait le "rescaling de vitesse"
    vy = ne.evaluate("(vy2+(Fy/(2*m)*dt))*(beta+1)") # pour controler la temperature
    #
    # fin des calculs utiles a Verlet
    fink = time.perf_counter()# marqueur fin temps de calcul
    print("Temps par pas : %s s"%(fink-debutk))
    pasCPU += [(fink-debutk)*1000] # stockage du temps de calcul
    tempstot += fink-debutk # temps de calcul total
    print("temps total: %s s"%(tempstot))
    #
# fin de la grosse boucle
###############################################
# fermeture des fichiers de positions
fichx.close()
fichy.close()
###############################################
### tous les dessins
# dessin final (dernieres positions)
plt.figure(2)
plt.xlim(XlimG,XlimD)
plt.ylim(YlimB,YlimH)
plt.plot(posx,posy,'ro', markersize=5)
plt.show()
# dessin du temps CPU
plt.figure(3)
plt.plot(pask,pasCPU)
plt.xlabel('pas')
plt.ylabel('temps de calcul (ms)')
plt.show()
# dessin de temperature
plt.figure(4)
line_T, = plt.plot(pastemps,pasT)
line_TC, = plt.plot(pastemps,pasTC)
plt.xlabel('temps (ps)')
plt.ylabel('Temperature (K)')
plt.legend([line_T, line_TC], ['T', 'T consigne'])
plt.show()
# dessin de MSD (mean square displacement)
plt.figure(5)
plt.plot(pasT,pasMSD)
plt.xlabel('Temperature (K)')
plt.ylabel('MSD')
plt.show()
# dessin de nbre liaison
plt.figure(6)
plt.plot(pasT,pasLiai)
plt.xlabel('Temperature K')
plt.ylabel('liaison par atome')
plt.show()
plt.figure(7)
plt.plot(pasEP,pasLiai)
plt.xlabel('Energie potentielle (J)')
plt.ylabel('liaison par atome')
plt.show()
# dessin des energies
plt.figure(8)
plt.subplot(211)
line_EC, = plt.plot(pask,pasEC)
plt.xlabel('pas')
plt.ylabel('Energies (J)')
plt.legend([line_EC], ['EC'])
plt.subplot(212)
line_EP, = plt.plot(pask,pasEP)
line_ET, = plt.plot(pask,pasET)
plt.xlabel('pas')
plt.ylabel('Energies (J)')
plt.legend([line_EP,line_ET], ['EP','ET'])
plt.show()
#
# film de la simu: image tout les X pas de temps
if film :
# ouverture pour la lecture
    fix=open('storex.npy','rb')
    fiy=open('storey.npy','rb')
    # liste de k ou tracer le graph
    klist = range(0,npas,pfilm)
    # boucle pour creer le film
    
    for k in range(npas):
        posx = np.load(fix) # on charge a chaque pas de temps
        posy = np.load(fiy) # on charge a chaque pas de temps
        # dessin a chaque pas (ne s'ouvre pas: est sauvegarde de maniere incrementale)
        if k in klist:
          # definition du domaine de dessin
          plt.ioff() # pour ne pas afficher les graphs)
          fig = plt.figure()
          plt.ylim(YlimB,YlimH)
          plt.xlim(XlimG,XlimD)
          plt.xlabel(k)
          plt.plot(posx,posy,'ro', markersize=5)
          fig.savefig('StorePic/MD-picture%s.png' % k) # sauvegarde incrementale
          plt.close(fig) # fermeture du graph
    # fin du film
    fix.close()
    fiy.close()
print ('%%%%%%%%%%%%%%%%%%%%%%')
# messages
print("nombre atomes: %s"%(npart))
print("nombre pas de temps: %s"%(npas))
print("temps simule: %s ps"%(dt*1e12*npas))
print("dimension max du systeme: %s Angstrom"%((max(nbrex,nbrey)-1)*re*1e10))
print (' ')