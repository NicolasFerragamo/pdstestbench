#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 21:59:46 2019

@author: nico
"""
import sys
sys.path.append('/home/nico/Documentos/facultad/6to_nivel/pds/git/pdstestbench')
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
SNR  = [-15, -8] #db
a1   = 2*np.sqrt(var)*pow(10,SNR[0]/20)
a2   = 2*np.sqrt(var)*pow(10,SNR[1]/20)
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

fftnoise = 10*np.log10(np.abs(fft(noise, axis=0)/N)**2)
#%% Gráficos de la PSD
jj = np.linspace(0, (N-1)*df, N)
plt.figure("PSD ruido", constrained_layout=True)
plt.title("PSD ruido")
plt.plot(jj, fftnoise, marker='.')
plt.xlabel('frecuecnia [rad]')
plt.ylabel("Amplitud [dB]")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

#%% generación de señales 
tt = np.linspace(0, (N-1)/fs, N)     
senoidal1 = np.vstack(np.transpose([a1 * np.sin(2*np.pi*j*tt) for j in f1]))  
senoidal2 = np.vstack(np.transpose([a2 * np.sin(2*np.pi*j*tt) for j in f1]))  
del tt   

signal1 = senoidal1 + noise
signal2 = senoidal2 + noise

Swelch1 = np.vstack(np.transpose([sp.welch(signal1[:,ii], L=L, over=over, win="Bartlett", ax=0) for ii in range(Nexp)]))
Swelch1 = 10*np.log10(Swelch1 *2/N)
Swelch1 = Swelch1 [:int(L/2),:]

Swelch2 = np.vstack(np.transpose([sp.welch(signal2[:,ii], L=L, over=over, win="Bartlett", ax=0) for ii in range(Nexp)]))
Swelch2 = 10*np.log10(Swelch2 *2/N)
Swelch2 = Swelch2 [:int(L/2),:]

freq_hat1 = np.argmax(Swelch1, axis=0)*fs/L # estimo la frecuecnia 
freq_hat2 = np.argmax(Swelch2, axis=0)*fs/L # estimo la frecuecnia 

error_freq1 = np.abs(freq_hat1 - f1)/f1  # calculo el error absoluto
error_medio1 = np.mean(error_freq1)

error_freq2 = np.abs(freq_hat2 - f1)/f1  # calculo el error absoluto
error_medio2 = np.mean(error_freq2)

varianza1_W = np.var(error_freq1)
varianza2_W = np.var(error_freq2)

#%% Gráficos de la PSD
ff = np.linspace(0, (N-1)*df/2, int(L/2))
plt.figure("PSD PSD con a1 = 3dB", constrained_layout=True)
plt.title("PSD con a1 = 3dB")
plt.plot(ff, Swelch1, marker='.')
plt.xlabel('frecuecnia [rad]')
plt.ylabel("Amplitud [dB]")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

# Gráfico del error absoluto
plt.figure("Error al estimar la frecuencia con a1 = 3dB", constrained_layout=True)
plt.title("Error al estimar la frecuencia")
plt.plot(f1, error_freq1,'*r')
plt.xlabel('frecuecnia [rad]')
plt.ylabel("Error relativo")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.xlim((min(f1), max(f1)))
plt.grid()


ff = np.linspace(0, (N-1)*df/2, int(L/2))
plt.figure("PSD con a1 = 10dB", constrained_layout=True)
plt.title("PSD con a1 = 10dB")
plt.plot(ff, Swelch2, marker='.')
plt.xlabel('frecuecnia [rad]')
plt.ylabel("Amplitud [dB]")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

# Gráfico del error absoluto
plt.figure("Error al estimar la frecuencia con a1 = 10dB", constrained_layout=True)
plt.title("Error al estimar la frecuencia")
plt.plot(f1, error_freq2,'*r')
plt.xlabel('frecuecnia [rad]')
plt.ylabel("Error relativo")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.xlim((min(f1), max(f1)))
plt.grid()


