#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:46:09 2019

@author: nico
"""


import os
import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
from scipy.fftpack import fft
os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 
import scipy.io as sio
from pdsmodulos.signals import spectral_estimation as sp

fs = 1000
##########################################
# Acá podés generar los gráficos pedidos #
##########################################
def vertical_flaten(a):
    
    return a.reshape(a.shape[0],1)

# para listar las variables que hay en el archivo
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])
N = len(ecg_one_lead)
hb_1 = vertical_flaten(mat_struct['heartbeat_pattern1'])

#%% obtengo la fft de la señal limpia
N1 = 35000
N2 = 90000
N3 = N2 - N1
df = fs/N3
signal = ecg_one_lead[N1:N2]

plt.figure("señal limpia")
plt.plot(signal)
plt.show()

K = 20
L = N3/K
over = 0.5
ff,Swelch = sig.welch(signal.flatten(),fs=fs,nperseg=L,window='bartlett')
#Swelch = sp.welch(signal.flatten(), L=L, over=over, win="Bartlett", ax=0)
Swelch = 10*np.log10(Swelch)
#Swelch = Swelch[:int(L/2)]

#ff = np.linspace(0, (N3-1)*df/2, int(L/2))
plt.figure("Señal limpia FFT")
plt.plot(ff,Swelch)
plt.xlabel('frecuecnia  [Hz]')
plt.ylabel('Amplitud db')
plt.grid()
plt.show()





#plt.figure(2)
#plt.plot(hb_1)
