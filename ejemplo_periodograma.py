#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:44:11 2019

@author: nico
"""

import os
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

N  = 1024 # muestras
fs = 2*np.pi # Hz
df = fs / N
a0 = 2 # Volts
p0 = 0 # radianes
f0 = np.pi / 2
Nexp = 200
mu = 0    # media (mu)
var = 2 # varianza

#%% generación de señales     
signal = np.vstack(np.transpose([ np.random.normal(0, np.sqrt(var), N)  for j in range(Nexp)]))

#%%  Grafico de los resultados
tt = np.linspace(0,N-1,N)
plt.figure("Gráfico de realizaciones de ruido blanco")
plt.plot(tt, signal)
plt.xlabel('tiempo [S]')
plt.ylabel('amplitud')
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.title("Gráfico dde realizaciones de ruido blanco")

#%% Periodograma
Sper = np.vstack(np.transpose([1/N *(np.abs(np.fft.fft(signal[:,ii])))**2 for ii in range(Nexp)]))

energia = np.sum(Sper, axis=0) / N

valor_medio_muestreal = np.mean(Sper, axis=1) 
valor_medio = np.mean(valor_medio_muestreal, axis=0) 

var_muestreal = np.var(Sper, axis=1) 
varianza =  np.mean(var_muestreal, axis=0)


#%%  Grafico de los resultados
ff = np.linspace(0,np.pi, N)
plt.figure("Gráfico promeio del periodograma para ruido blanco")
plt.subplot(211)
plt.plot(ff,valor_medio_muestreal)
plt.xlabel('frecuecnia normalizada  [Rad]')
plt.ylabel('valor medio')
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.title("Valor medio muestral")
plt.subplot(212)
plt.plot(ff,var_muestreal)
plt.xlabel('frecuecnia normalizada  [Rad]')
plt.ylabel('varianza')
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.title("varianza muestral")
plt.tight_layout() #para ajustar el tamaño de lo contrario se puperpinan los titulos

