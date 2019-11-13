#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:20:39 2019

@author: nico
"""

from spectrum import CORRELOGRAMPSD
import os
import matplotlib.pyplot as plt
import numpy as np
from pdsmodulos.signals import spectral_estimation as sp
import pandas as pd

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

# Simular para los siguientes tamaños de señal
N    = 1000
Nexp = 200
fs   = 2*np.pi # Hz
df   = fs/N
f0   = np.pi / 2
M    = 100
mu   = 0    # media (mu)
var  = 2 # varianza
SNR  = -15 #db
a1   = 2*np.sqrt(var)*pow(10,SNR/20)
a    = (-1/2) *df
b    = (1/2) * df

#%% generación de frecuencias aleatorias
fa = np.random.uniform(a, b, size = (Nexp)) # genera aleatorios
f1 = f0 + fa   

plt.hist(fa, bins=10, alpha=1, edgecolor = 'black',  linewidth=1)
plt.ylabel('frequencia')
plt.xlabel('valores')
plt.title('Histograma Uniforme')
plt.savefig("Histograma.png")
plt.show()

del fa


#%% generación de señales    
noise= np.vstack(np.transpose([ np.random.normal(0, np.sqrt(var), N)  for j in range(Nexp)]))

#%% generación de señales 
tt = np.linspace(0, (N-1)/fs, N)     
senoidal = np.vstack(np.transpose([a1 * np.sin(2*np.pi*j*tt) for j in f1]))  
del tt   

signal = senoidal + noise

Swelch = np.vstack(np.transpose([CORRELOGRAMPSD(signal[:,ii],window='bartlett', lag=M, NFFT=N) for ii in range(Nexp)]))
#Swelch = np.vstack(np.transpose([sp.blakmanTukey(signal[:,ii], win="Bartlett", M=M,ax=0) for ii in range(Nexp)]))
Swelch = 10*np.log10(Swelch *2/N)
#Swelch = Swelch [:int(L/2),:]

freq_hat = np.argmax(Swelch[:int(N/2)-1,:], axis=0)*df # estimo la frecuecnia 

error_freq = np.abs(freq_hat - f1)/f1  # calculo el error absoluto

#%% Gráficos de la PSD
ff = np.linspace(0, (N-1)*df, N)
plt.figure("PSD", constrained_layout=True)
plt.title("PSD")
plt.plot(ff, Swelch, marker='.')
plt.xlabel('frecuecnia [rad]')
plt.ylabel("Amplitud [dB]")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

# Gráfico del error absoluto
plt.figure("Error al estimar la frecuencia", constrained_layout=True)
plt.title("Error al estimar la frecuencia")
plt.plot(f1, error_freq,'*r')
plt.xlabel('frecuecnia [rad]')
plt.ylabel("Error relativo")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.xlim((min(f1), max(f1)))
plt.grid()