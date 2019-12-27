#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:11:44 2019

@author: nico
"""
import sys
sys.path.append('/home/nico/Documentos/facultad/6to_nivel/pds/git/pdstestbench')
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from pdsmodulos.signals import FFT
from pdsmodulos.signals import windows as wds
from pdsmodulos.signals import signals as sig
from scipy.fftpack import fft

#import seaborn as sns

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

N  = 1000 # muestras
fs = 2*np.pi # Hz
df = fs / N
a0 = 1 # Volts
f0 = np.pi / 2

f1 = f0 + df*np.array([0.01, 0.25, 0.5],float)
fd1 = np.array(['0.01', '0.25', '0.5'])
f2 = f0+ 10*df
A1 = np.array([a0 * 10**(-50/20), a0 * 10**(-25/20), a0 * 10**(-30/20)])
A  = np.array(['a1 = -50 dB', 'a1 = -25 dB', 'a1 = -30 dB'])

#%% Señales ej 2c
tt = np.linspace(0, (N-1)/fs, N)  
freq = np.linspace(0, (N-1)*df, N) / fs
V = len(f1)
signal = np.vstack(np.transpose([(a0 * np.sin(2*np.pi*f1[j]*tt) + A1[j] * np.sin(2*np.pi*f2*tt))   for j in range(V)])) 

mod_signal = np.vstack(np.transpose([np.abs(fft(signal[:,ii]))*2/N  for ii in  range(V)]))
mod_signal = 20 *np.log10(mod_signal)

for (ii, this_fd1) in zip(range(V), fd1):
# grafico
     fig = plt.figure("Señal bitonal separada " + this_fd1, constrained_layout=True)
     plt.title("Señal bitonal separada " + this_fd1)
     plt.plot(freq[0:int(N/2)],mod_signal[0:int(N/2), ii], label=fd1[ii])
     plt.xlabel("Frecuencia normalizada [f/fs]")
     plt.ylabel("Magnitud [dB]")
     plt.axhline(0, color="black")
     plt.axvline(0, color="black")
     plt.legend(loc = 'upper right')
     plt.text(0.02, -8, A[ii], style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
     plt.grid()