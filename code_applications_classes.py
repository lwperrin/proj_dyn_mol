import matplotlib.pyplot as plt
import numpy as np
from code_1 import *

r = 71*10**-12
sig = 3.4*10**-10
rmin=2**(1/6)*sig
rcut = 2**(7/6)*sig
print(rmin)
E0 = 1.65*10**-21

atome_ref = Atome(0,0,0,r)

X = np.linspace(10**-11,30*10**-10,5000) # petit problème de distances...
atome_mobile = [Atome(x,0,0,r) for x in X] #on crée un atome mobile pour observer la variation d'énergie potentielle et de force en fonction de la distance

liste_inter = [Interaction(atome_ref,atome_mobile[i],rcut) for i in range(len(atome_mobile))] #liste des interactions
EP = [liste_inter[i].lennardjones_potentiel(E0,sig) for i in range(len(liste_inter))]

plt.plot(X,EP)
plt.plot([rmin],[0],'*')
plt.show()
for i in atome_mobile[200:220]:
    atome_ref.affiche()
    i.affiche()
    plt.show()