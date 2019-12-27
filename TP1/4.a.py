#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 17:35:36 2019

@author: nico
"""

#%% importo los paquetes necesarios
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
f0 = 9*df# Hz

#%% Genero las variables necesarias

signal    = np.zeros(N)
fftsignal = np.zeros(N, complex)

energia_temporal = 0
energia_frecuencia = 0
energia_frecuencia_puntual = 0
energia_max_frecuencia = 0
#%% generacion y muestreo de las senoidal

tt, signal = sg.seno(fs, f0, N, a0, p0)
fftsignal    = np.fft.fft(signal)
mod_fftsignal = np.abs(fftsignal)


#%% Gŕaficos de las señales en tiempo y en frecuenciales
plt.figure("Gráfico de la señal temporal")
plt.plot(tt, signal, 'b', label='f0 = 9d.f')
plt.xlabel('tiempo [S]')
plt.ylabel('Amplitud [UA]')
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.title('Gráfico de la señal temporal')
plt.legend(loc = 'upper right')

FFT.plotFFT(fftsignal, fs, N, y1l='Amplitud Normlizada [db] ', y2l='Fase [rad] ',
              c=0, db='ON', tipo='plot', m='.')
#%%  Cálculo de la energía

# energía temporal
for ii in range(0,N):
     energia_temporal += signal[ii]**2

energia_temporal /= N
print ("la energía termporal es: " ,energia_temporal)

#energia frecuencial
for ii in range(0,N):
     energia_frecuencia += mod_fftsignal[ii]**2
     
energia_frecuencia = energia_frecuencia/N**2
print ("la energía por frecuencia es: " ,energia_frecuencia)


# Energía puntual
mod_fftsignal1 = np.zeros(int(N/2))
mod_fftsignal1 = mod_fftsignal[:int(N/2)]

max_signal = np.amax(mod_fftsignal1, axis=0)


for jj in range(0, int(N/2)) :
     if (max_signal == mod_fftsignal1[jj]) :
          k = jj
          
print('la frecuencia estimada es: ', k)


energia_frecuencia_puntual = mod_fftsignal1[9] **2
energia_frecuencia_puntual = energia_frecuencia_puntual *2/(N**2)
print('la energía estimada puntual es: ', energia_frecuencia_puntual)

energia_max_frecuencia = max_signal **2
energia_max_frecuencia = energia_max_frecuencia *2/(N**2)
print('la energía estimada maxima es: ', energia_max_frecuencia)

#%% Relleno de tabla 

prediccion = ['0.5', '0,5', '0,5']
resultados = [energia_frecuencia, energia_frecuencia_puntual, energia_max_frecuencia]

tus_resultados = [ ['$\sum_{f=0}^{f_S/2} \lvert X(f) \rvert ^2$', '$ \lvert X(f_0) \rvert ^2 $', '$ \mathop{arg\ max}_f \{\lvert X(f) \rvert ^2\} $'], 
                   ['',                                     '',                           '$f \in [0:f_S/2]$'], 
                  ['', '', ''], 
                  [prediccion[0], prediccion[1], prediccion[2]], # <-- completar acá
                  ['', '', ''], 
                  [resultados[0], resultados[1], resultados[2]]  # <-- completar acá
                 ]
df = pd.DataFrame(tus_resultados, columns=['Energía total', 'Energía en $f_0$', 'Máximo de Energía'],
               index=['$f_0$ \ expr. matemática', 
                      '', 
                      '', 
                      'predicción', 
                      '', 
                      'simulación'])

print(df)