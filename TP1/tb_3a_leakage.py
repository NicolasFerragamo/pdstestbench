#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:05:05 2019

@author: nico
"""
#%% importo los paquetes necesarios
import os
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('/home/nico/Documentos/facultad/6to_nivel/pds/git/pdstestbench')
from pdsmodulos.signals import signals as sg 
from pdsmodulos.signals import FFT

#%% limpio el entorno
os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

#%% Estalesco los datos necesarios
N  = 1000 # muestras
fs = 1000 # Hz
df = fs / N
a0 = 1 # Volts
p0 = 0 # radianes
f0 = fs / 4 # Hz

#%% Genero las variables necesarias

fd  = np.array([0, 0.01*df, 0.25*df, 0.5*df],float)
fd1 = np.array(['0', '0.01 df', '0.25 df', '0.5 df'])
f   = f0 + fd
del fd

signal    = np.zeros((N, 4))
fftsignal = np.zeros((N, 4), complex)
modfftsignal = np.zeros((N, 4), float)
fasefftsignal = np.zeros((N, 4), float)
energia = np.zeros((4))
#%% generacion y muestreo de las senoidal

for ii in range(0, 4):
    tt, signal[:,ii] = sg.seno(fs, f[ii], N, a0, p0)
  
energ0 = np.sum(signal**2, axis=0)/N
#%% Gráficos de las señales temporales
#ax = plt.figure("Funcion  senoidal")
#plt.plot(tt, signal[:,0], color='blue',label='sin(2pi(f0+0.01df)t)')
#plt.plot(tt, signal[:,1], color='red',label='sin(2pi(f0+0.25df)t)')
#plt.plot(tt, signal[:,2], color='green',label='sin(2pi(f0+0.5df)t)')
#plt.xlabel('tiempo [segundos]')
#plt.ylabel('Amplitud [UA] ')
#plt.axhline(0, color="black")
#plt.axvline(0, color="black")
#plt.grid()
#plt.title('Funcion senoidal')
#plt.legend(loc = 'upper right')
#plt.show()

#%% Gŕaficos de las señales frecuenciales

for ii in range(0, 4) : # no incluye el 4
    fftsignal[:,ii] = np.fft.fft(signal[:,ii])
    FFT.plotFFT(fftsignal[:,ii],fs,N, tp= 'FFT', c=ii, l=fd1[ii], db='on', m='.', ls='None')
    
#%% obtengo los datos del módulo para las tablas
for ii in range(0, 4) :
     modfftsignal[:, ii] = np.abs(fftsignal[:, ii])
     
     
for ii in range(0,4) :
     for jj in range(0,N) :
          if jj != f0 :
               energia[ii] += modfftsignal[jj,ii]**2

     
energia = energia / (N**2)   

import pandas as pd
tus_resultados = [ ['$ \lvert X(f_0) \lvert$', '$ \lvert X(f_0+1) \lvert $', '$\sum_{i=F} \lvert X(f_i) \lvert ^2 $'], 
                   ['',                        '',                           '$F:f \neq f_0$'], 
                  [modfftsignal[250,0]/N, modfftsignal[251,0]/N, energia[0]], # <-- acá debería haber numeritos :)
                  [modfftsignal[250,1]/N, modfftsignal[251,1]/N, energia[1]], # <-- acá debería haber numeritos :)
                  [modfftsignal[250,2]/N, modfftsignal[251,2]/N, energia[2]], # <-- acá debería haber numeritos :)
                  [modfftsignal[250,3]/N, modfftsignal[251,3]/N, energia[3]]  # <-- acá debería haber numeritos :)
                 ]
df = pd.DataFrame(tus_resultados, columns=['Frecuencia central', 'Primer adyacente', 'Resto de      frecuencias'],index=['$f_0$ \ expr. matemática', 
                      '', 
                      '$f_S/4$', 
                      '$f_S/4+0.01$', 
                      '$f_S/4+0.25$', 
                      '$f_S/4+0.5$'])

print(df)