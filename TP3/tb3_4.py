#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 21:59:46 2019

@author: nico
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
#import seaborn as sns
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
over = 0.5
mu   = 0    # media (mu)
var  = 2 # varianza
K    = 10
L    = N/K
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

#fftnoise = 10*np.log10(np.abs(fft(noise, axis=0)/N)**2)
##%% Gráficos de la PSD
#jj = np.linspace(0, (N-1)*df, N)
#plt.figure("PSD ruido", constrained_layout=True)
#plt.title("PSD ruido")
#plt.plot(jj, fftnoise, marker='.')
#plt.xlabel('frecuecnia [rad]')
#plt.ylabel("Amplitud [dB]")
#plt.axhline(0, color="black")
#plt.axvline(0, color="black")
#plt.grid()
#%% generación de señales 
tt = np.linspace(0, (N-1)/fs, N)     
senoidal = np.vstack(np.transpose([a1 * np.sin(2*np.pi*j*tt) for j in f1]))  
del tt   

signal = senoidal + noise

Swelch = np.vstack(np.transpose([sp.welch(signal[:,ii], L=L, over=over, win="Bartlett", ax=0) for ii in range(Nexp)]))
Swelch = 10*np.log10(Swelch *2/N)
Swelch = Swelch [:int(L/2),:]

freq_hat = np.argmax(Swelch, axis=0)*fs/L # estimo la frecuecnia 

error_freq = np.abs(freq_hat - f1)/f1  # calculo el error absoluto

#%% Gráficos de la PSD
ff = np.linspace(0, (N-1)*df/2, int(L/2))
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