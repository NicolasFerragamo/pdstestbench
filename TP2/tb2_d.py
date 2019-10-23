#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:51:41 2019

@author: nico
"""

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

f1 = f0 + df*np.array([0.01, 0.25, 0.5],float)
fd1 = np.array(['0.01', '0.25', '0.5'])
f2 = f0+ 10*df

A1 = np.array([a0 * 10**(-265/20), a0 * 10**(-35/20), a0 * 10**(-90/20), a0*10**(-90/20), a0*10**(-100/20)])
A  = np.array(['a1 = -265 dB', 'a1 = -35 dB', 'a1 = -90 dB','a1 = -90 dB', 'a1 = -100 dB'])

B1 = np.array([a0 * 10**(-30/20), a0 * 10**(-50/20), a0 * 10**(-60/20), a0*10**(-70/20), a0*10**(-85/20)])
B  = np.array(['a1 = -30 dB',     'a1 = -50 dB',     'a1 = -60 dB',    'a1 = -70 dB',   'a1 = -85 dB'])

ventanas = [sg.boxcar(N), np.bartlett(N), np.hanning(N), np.blackman(N),  sg.flattop(N)]
V =  len(ventanas)
ventana = ["rectangular",'Barlett',"hanning", "blackman",  "flattop"]

# genero los ejes de tiempo y frecuencia
tt = np.linspace(0, (N-1)/fs, N)  
freq = np.linspace(0, (N-1)*df, N) / fs
#%% Señales ej 2d-b

for (ii, this_w) in zip(range(V), ventanas):
     
     signal =(a0 * np.sin(2*np.pi*f0*tt) + A1[ii] * np.sin(2*np.pi*f2*tt)) * this_w 
     
     mod_signal = np.abs(fft(signal))*2/N  
     mod_signal = 20 *np.log10(mod_signal)
     
     # grafico
     fig = plt.figure("Señal bitonal separada 10df con ventana " + ventana[ii], constrained_layout=True)
     plt.title("Señal bitonal separada " + ventana[ii])
     plt.plot(freq[0:int(N/2)],mod_signal[0:int(N/2)], label=ventana[ii])
     plt.xlabel("Frecuencia normalizada [f/fs]")
     plt.ylabel("Magnitud [dB]")
     plt.axhline(0, color="black")
     plt.axvline(0, color="black")
     plt.legend(loc = 'upper right')
     plt.text(0.02, -15, A[ii], style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
     plt.grid()



#%% Señales ej 2d-c ahora tenemos desintonia maxima
     
for (ii, this_w) in zip(range(V), ventanas):
     
     signal =(a0 * np.sin(2*np.pi*f1[2]*tt) + B1[ii] * np.sin(2*np.pi*f2*tt)) * this_w 
     
     mod_signal = np.abs(fft(signal))*2/N  
     mod_signal = 20 *np.log10(mod_signal)
     
     # grafico
     fig = plt.figure("Señal bitonal separada 10df, desinonizada 0,5df con ventana " + ventana[ii], constrained_layout=True)
     plt.title("Señal bitonal separada 10df, desinonizada 0,5df con ventana " + ventana[ii])
     plt.plot(freq[0:int(N/2)],mod_signal[0:int(N/2)], label=ventana[ii])
     plt.xlabel("Frecuencia normalizada [f/fs]")
     plt.ylabel("Magnitud [dB]")
     plt.axhline(0, color="black")
     plt.axvline(0, color="black")
     plt.legend(loc = 'upper right')
     plt.text(0.02, -15, B[ii], style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
     plt.grid()
     
     

#%% data frame

tus_resultados = np.vstack(np.transpose([A,B]))
df = pd.DataFrame(tus_resultados, columns=['$a^0_2$ (dB)','$a^1_2$ (dB)'],
               index=[  
                        'Rectangular',
                        'Bartlett',
                        'Hann',
                        'Blackman',
                        'Flat-top'
                     ])  
print(df)   