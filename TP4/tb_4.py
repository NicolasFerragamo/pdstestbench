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

plt.figure("señal principal")
plt.plot(ecg_one_lead)
plt.show()


#%% obtengo la fft de la señal limpia sin ruido esto me sirbe para obtener los 
# las fs1 y fp1

N1 = 35000
N2 = 90000
N3 = N2 - N1
df = fs/N3
signal = ecg_one_lead[N1:N2]

plt.figure("señal limpia")
plt.plot(signal)
plt.show()

K = 10
L = N3/K
over = 0.5
ff,Swelch = sig.welch(signal.flatten(),fs=fs,nperseg=L,window='bartlett')
#Swelch = sp.welch(signal.flatten(), L=L, over=over, win="Bartlett", ax=0)
Swelch = 10*np.log10(Swelch)
#Swelch = Swelch[:int(L/2)]

#ff = np.linspace(0, (N3-1)*df/2, int(L/2))
plt.figure("Señal limpia FFT")
plt.title(" Con esto obtengo fp1 y fs1")
plt.plot(ff,Swelch)
plt.xlabel('frecuecnia  [Hz]')
plt.ylabel('Amplitud db')
plt.grid()
plt.show()



#%% obtengo la fft de un latido limpio para obtener wp2 
N4 = 17750
N5 = 18690
N6 = N5 - N4
df1 = fs/N6
signal1 = ecg_one_lead[N4:N5]

plt.figure("Latido limpio")
plt.plot(signal1)
plt.show()

K1 = 10
L1 = N6/K
ff1,Swelch1 = sig.welch(signal1.flatten(),fs=fs,nperseg=L1,window='bartlett')
Swelch1 = 10*np.log10(Swelch1/Swelch1[0])

plt.figure("Latido limpio FFT")
plt.title(" Con esto obtengo wp2")
plt.plot(ff1,Swelch1)
plt.xlabel('frecuecnia  [Hz]')
plt.ylabel('Amplitud db')
plt.grid()
plt.show()


#%% obtengo la fft de ruido para obtener ws2 
N7 = 106340
N8 = 106840
N9 = N8 - N7
df2 = fs/N9
signal2 = ecg_one_lead[N7:N8]

plt.figure("Ruido en el electro")
plt.plot(signal2)
plt.show()

K2 = 10
L2 = N9/K2
ff2,Swelch2 = sig.welch(signal2.flatten(),fs=fs,nperseg=L1,window='bartlett')
Swelch2 = 10*np.log10(Swelch2)

plt.figure("Ruido en el electro FFT")
plt.title(" Con esto obtengo fs2")
plt.plot(ff2,Swelch2)
plt.xlabel('frecuecnia  [Hz]')
plt.ylabel('Amplitud db')
plt.grid()
plt.show()
