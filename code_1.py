from math import *
import numpy as np
import matplotlib.pyplot as plt


class Atome :
	#On définit un atome par les coordonnées dans l'espace de son centre et son rayon.
	def __init__(self, x,y,z,r):
		self.x=x #position du centre de l'atome dans l'espace
		self.y=y
		self.z=z
		self.r=r #rayon de l'atome

	def __mul__(self,other): #changement de la taille de l'atome, pour changement de matériau par exemple
		if type(other)==int or type(other)==float :
			res0,res1,res2,res3=self.x,self.y,self.z,other*self.r
		else :
			raise NotImplemented
		return Atome(res0,res1,res2,res3)

	def __sub__(self,other): #renvoie la distance entre 2 atomes
		assert type(other)==Atome
		dist=sqrt((self.x-other.x)**2+(self.y-other.y)**2+(self.z-other.z**2))
		return dist

	def affiche(self): #affiche l'atome et sa position
		liste_x=[self.r*cos(t*10**-2)+self.x for t in range(0, 700)]
		liste_y=[self.r*sin(t*10**-2)+self.y for t in range(0, 700)]
		plt.plot(self.x, self.y, '*')
		plt.plot(liste_x,liste_y)
		return None

class Interaction :
	def __init__(self,atome1,atome2,rcut):
		self.atome1=atome1 #premier atome
		self.atome2=atome2 #2eme atome
		self.rcut=rcut #distance de coupure
		self.dist=atome1-atome2
	
	def lennardjones_potentiel(self,E0,sig): #energie potentielle d'interacion
		potentiel=4*E0*((sig/self.dist)**12-(sig/self.dist)**6)
		return potentiel

	def lennardjones_force(self,E0,sig): #force d'interaction entre les atomes
		force=-4*E0*(6*sig**6/self.dist**7-12*sig**12/self.dist**13)
		return force