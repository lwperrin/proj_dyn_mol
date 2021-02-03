from code_1 import *
from random import *
import matplotlib.pyplot as plt

#paramètres atomes
m=6.69*10**-26
r = 71*10**-12
sig = 3.4*10**-10
rmin=2**(1/6)*sig
rcut = 2**(7/6)*sig
E0 = 1.65*10**-21
nb_atms=10

liste_positions=[[randint(-100,100)/10**11, randint(-100,100)/10**11] for k2 in range(nb_atms)]
liste_atomes=[Atome(liste_positions[i][0],liste_positions[i][1],0,r) for i in range(len(liste_positions))]
Matrice_interactions=[[0 for i in range(nb_atms)] for j in range(nb_atms)]

#paramètres simulation
dt=0.01
tfin=2
for atome in liste_atomes :
    atome.affiche()
plt.show()
plt.close()
liste_atomes2 = liste_atomes[:]

for t in range(int(tfin/dt)):
    plt.figure(figsize=(16, 9))
    for atome in range(nb_atms): #On calcule la matrice d'interactions à un instant
        for autre in range(nb_atms):
            Matrice_interactions[atome][autre]=Interaction(liste_atomes[atome],liste_atomes[autre],rcut)
        liste_interactions=Matrice_interactions[atome]
        x, y = liste_atomes[atome].xpos(), liste_atomes[atome].ypos()
        x2, y2 = liste_atomes2[atome].xpos(), liste_atomes2[atome].ypos()
        accx, accy = 0, 0
        for i in range(nb_atms):
            force = liste_interactions[i].lennardjones_force(E0,sig)
            if force != 0:
                accx, accy = accx+m*force*(liste_atomes[i].xpos()-liste_atomes[atome].xpos())/(liste_atomes[i]-liste_atomes[atome]), accy+m*force*(liste_atomes[i].ypos()-liste_atomes[atome].ypos())/(liste_atomes[i]-liste_atomes[atome])
        x, x2 = 2*x-x2+0.5*accx*dt**2, x
        y, y2 = 2*y-y2+0.5*accy*dt**2, y
        liste_atomes[atome] = Atome(x,y,0,r)
        liste_atomes2[atome] = Atome(x2,y2,0,r)
        liste_atomes[atome].affiche()
        liste_atomes[atome].afficher_force([accx/m,accy/m])
    plt.show()
    #plt.savefig("C:/Users/lwper/OneDrive/Bureau/proj2b/partagé/proj_dyn_mol/images_10_atomes/image_" + str(t) + ".pdf")
    plt.close()