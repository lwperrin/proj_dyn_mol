import matplotlib.pyplot as plt
from code_1 import *

r=71*10**-12
sig=3.4*10**-10
rcut=2**(7/6)*sig
E0=1.65*10**-21

atome_ref=Atome(0,0,0,r)

X=[k*10**-14 for k in range(1,1000)]
atome_mobile=[Atome(x,0,0,r) for x in X] #on crée un atome mobile pour observer la variation d'énergie potentielle et de force en fonction de la distance

liste_inter=[Interaction(atome_ref,atome_mobile[i],rcut) for i in range(len(atome_mobile))] #liste des interactions
EP=[liste_inter[i].lennardjones_potentiel(E0,sig) for i in range(len(liste_inter))]

plt.plot(X,EP)
plt.show()
atome_ref.plot()
plt.show()