#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:30:33 2019

@author: nico
"""
import sys
sys.path.append('/home/nico/Documentos/facultad/6to_nivel/pds/git/pdstestbench')
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg
from pdsmodulos.signals import FFT
from pdsmodulos.signals import windows as wds
from pdsmodulos.signals import signals as sig
from scipy.fftpack import fft
import pandas as pd

#import seaborn as sns

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

N  = 1000 # muestras
fs = 2*np.pi # Hz
df = fs / N
a0 = 1 # Volts
f0 = np.pi / 2
f1 = f0 + 0.5* df
f2 = np.array([f0 + 16*df, f0 + 6*df, f0 + 5*df, f0 + 5*df, f0 + 7*df,])
f2list =      ['16 df',    '6 df',    '5 df',   '5df',      '7 df']
dif = -40 #dB
a1  = a0 * 10**(dif/20)



ventanas = [sg.boxcar(N), np.bartlett(N), np.hanning(N), np.blackman(N),  sg.flattop(N)]
V =  len(ventanas)
ventana = ["rectangular",'Barlett',"hanning", "blackman",  "flattop"]

# genero los ejes de tiempo y frecuencia
tt = np.linspace(0, (N-1)/fs, N)  
freq = np.linspace(0, (N-1)*df, N) / fs
#%% Señales ej 2d-b

for (ii, this_w) in zip(range(V), ventanas):
     
     signal =(a0 * np.sin(2*np.pi*f1*tt) + a1 * np.sin(2*np.pi*f2[ii]*tt)) * this_w 
     
     mod_signal = np.abs(fft(signal))*2/N  
     mod_signal = 20 *np.log10(mod_signal)
     
     # grafico
     fig = plt.figure("Señal bitonal separada " + f2list[ii] + " con ventana " + ventana[ii], constrained_layout=True)
     plt.title("Señal bitonal separada " + f2list[ii] + " con ventana " + ventana[ii])
     #plt.plot(freq[0:int(N/2)],mod_signal[0:int(N/2)], label=ventana[ii], marker='.', linestyle='None')
     plt.plot(freq[0:int(N/2)],mod_signal[0:int(N/2)], label=ventana[ii])
     plt.xlabel("Frecuencia normalizada [f/fs]")
     plt.ylabel("Magnitud [dB]")
     plt.axhline(0, color="black")
     plt.axvline(0, color="black")
     plt.legend(loc = 'upper right')
     plt.text(0.02, -15, f2list[ii], style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
     plt.grid()
     
     

#%% data frame


tus_resultados = f2list
df = pd.DataFrame(tus_resultados, columns=['$\Omega_0$ (#)'],
               index=[  
                        'Rectangular',
                        'Bartlett',
                        'Hann',
                        'Blackman',
                        'Flat-top'
                     ])
print(df)   