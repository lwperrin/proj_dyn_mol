import matplotlib.pyplot as plt
import numpy as np
from code_1 import *

r = 71*10**-12
sig = 3.4*10**-10
rmin=2**(1/6)*sig
rcut = 2**(7/6)*sig
E0 = 1.65*10**-21
atome_ref = Atome(0,0,0,r)

X = np.linspace(3.05*10**-10, 1.5*10**-9, 500) # petit problème de distances...
atome_mobile = [Atome(x,0,0,r) for x in X] #on crée un atome mobile pour observer la variation d'énergie potentielle et de force en fonction de la distance

liste_inter = [Interaction(atome_ref,atome_mobile[i],rcut) for i in range(len(atome_mobile))] #liste des interactions
EP = [liste_inter[i].lennardjones_potentiel(E0,sig) for i in range(len(liste_inter))]
F = [liste_inter[i].lennardjones_force(E0,sig) for i in range(len(liste_inter))]

# energie potentielle
plt.plot(X,EP)
plt.plot([rmin],[-1.65*10**-21],'*')
axes=plt.gca()
axes.set_xlim(0,1.5*10**-9)
axes.set_ylim(-0.00002*10**-16,0.0001*10**-16)
plt.show()


# position des atomes
for i in range(len(atome_mobile)):
    plt.figure(figsize=(16,9))
    plt.subplot(1,2,1)
    atome_ref.affiche()
    atome_mobile[i].affiche()
    atome_mobile[i].afficher_force([F[i], 0, 0])
    atome_ref.afficher_force([-F[i], 0, 0])
    plt.legend(["centre atome", "atome", "centre deuxième atome", "deuxième atome", "vecteur d'effort"], loc='upper right')

    plt.subplot(1,2,2)
    plt.plot(X, EP)
    plt.plot(atome_mobile[i]-atome_ref, EP[i], '*',color='r')
    axes = plt.gca()
    axes.set_xlim(0, 1.5 * 10 ** -9)
    axes.set_ylim(-0.00002 * 10 ** -16, 0.0001 * 10 ** -16)
    plt.xlabel('distance entre les atomes en m')
    plt.ylabel("Energie potentielle d'interaction en J")
    #plt.show()
    plt.savefig("C:/Users/lwper/OneDrive/Bureau/proj2b/partagé/proj_dyn_mol/images_2_atomes/image_"+str(i)+".pdf")