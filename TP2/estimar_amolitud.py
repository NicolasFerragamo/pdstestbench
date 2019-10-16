#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 19:31:47 2019

@author: nico
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import statistics as stats
import scipy.signal as sg

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

N  = 1000 # muestras
fs = 2*np.pi # Hz
df = fs / N
a0 = 2 # Volts
p0 = 0 # radianes
f0 = np.pi / 2
Nexp = 500

a = -2 * df
b = 2 * df

#%% generación de frecuencias aleatorias
fa = np.random.uniform(a, b, size = (Nexp)) # genera aleatorios

plt.hist(fa, bins=20, alpha=1, edgecolor = 'black',  linewidth=1)
plt.ylabel('frequencia')
plt.xlabel('valores')
plt.title('Histograma Uniforme')
plt.savefig("Histograma.png")
plt.show()

#%% generación de señales
f1 = f0 + fa
del fa     

ventanas = [np.hanning(N), np.hamming(N), np.blackman(N), sg.boxcar(N), sg.flattop(N)]
V =  len(ventanas)
ventana = ["hanning", "hamming", "blackman", "rectangular", "flattop"]
sesgo = np.zeros((V))
a_est = np.zeros((Nexp, V))
a_mean = np.zeros((V))
varianza = np.zeros((V))
tt = np.linspace(0, (N-1)/fs, N)     

for (ii, this_w) in zip(range(V), ventanas):
     signal = np.vstack(np.transpose([a0 * np.sin(2*np.pi*j*tt) * this_w  for j in f1]))  
    
     mod_signal = np.vstack(np.transpose([np.abs(np.fft.fft(signal[:,ii]))*2/N  for ii in      range(Nexp)]))

     mod_signal = mod_signal[0:int(N/2)]

     a_est[:,ii] = mod_signal[int(N/4)]

     a_mean[ii] = stats.mean(a_est[:,ii])

     sesgo[ii] = a_mean[ii] - a0

     varianza[ii] = stats.variance(a_est[:, ii])
     
     plt.figure("Histograma de a_est con ventana "  )
     plt.hist(a_est[:,ii], bins=20, alpha=1, edgecolor = 'black',  linewidth=1, label=ventana[ii])
     plt.legend(loc = 'upper right')
     plt.ylabel('frecuencia')
     plt.xlabel('valores')
     plt.title('histograma de errores al estimar amplitud')
     plt.show()    

