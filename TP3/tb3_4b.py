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
from scipy.fftpack import fft
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

S_BT1 = np.vstack(np.transpose([CORRELOGRAMPSD(signal1[:,ii],window='bartlett', lag=M, NFFT=N) for ii in range(Nexp)]))
S_BT1 = 10*np.log10(S_BT1 *2/N)

S_BT2 = np.vstack(np.transpose([CORRELOGRAMPSD(signal2[:,ii],window='bartlett', lag=M, NFFT=N) for ii in range(Nexp)]))
S_BT2 = 10*np.log10(S_BT2 *2/N)


freq_hat1 = np.argmax(S_BT1[:int(N/2)-1,:], axis=0)*df # estimo la frecuecnia 
freq_hat2 = np.argmax(S_BT2[:int(N/2)-1,:], axis=0)*df # estimo la frecuecnia 

error_freq1 = np.abs(freq_hat1 - f1)/f1  # calculo el error absoluto
error_freq2 = np.abs(freq_hat2 - f1)/f1  # calculo el error absoluto

error_medio1 = np.mean(error_freq1)
error_medio2 = np.mean(error_freq2)

varianza1_BT = np.var(error_freq1)
varianza2_BT = np.var(error_freq2)

#%% Gráficos de la PSD
ff = np.linspace(0, (N-1)*df, N)
plt.figure("PSD con a1 = 3dB", constrained_layout=True)
plt.title("PSD con a1 = 3dB")
plt.plot(ff, S_BT1, marker='.')
plt.xlabel('frecuecnia [rad]')
plt.ylabel("Amplitud [dB]")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

# Gráfico del error absoluto
plt.figure("Error al estimar la frecuencia con a1 = 3dB", constrained_layout=True)
plt.title("Error al estimar la frecuencia con a1 = 3dB")
plt.plot(f1, error_freq1,'*r')
plt.xlabel('frecuecnia [rad]')
plt.ylabel("Error relativo")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.xlim((min(f1), max(f1)))
plt.grid()


plt.figure("PSD con a1 = 10dB", constrained_layout=True)
plt.title("PSD con a1 = 10dB")
plt.plot(ff, S_BT2, marker='.')
plt.xlabel('frecuecnia [rad]')
plt.ylabel("Amplitud [dB]")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

# Gráfico del error absoluto
plt.figure("Error al estimar la frecuencia con a1 = 10dB", constrained_layout=True)
plt.title("Error al estimar la frecuencia con a1 = 10dB")
plt.plot(f1, error_freq2,'*r')
plt.xlabel('frecuecnia [rad]')
plt.ylabel("Error relativo")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.xlim((min(f1), max(f1)))
plt.grid()