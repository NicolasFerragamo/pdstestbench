#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:35:33 2019

@author: nico
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from pdsmodulos.signals import FFT
from pdsmodulos.signals import windows as wds
from pdsmodulos.signals import signals as sig

#import seaborn as sns

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

N  = 1000 # muestras
fs = 2*np.pi # Hz
df = fs / N
a0 = 1 # Volts
p0 = 0 # radianes
f0 = np.pi / 2
dif= -40  #Diferencia en dB de Amplitude
a1 = a0 * 10**(dif/20)
f1 = f0 + 10 * df

#%% Señales ej 2a
tt = np.linspace(0, (N-1)/fs, N)  
freq = np.linspace(0, (N-1)*df, N) / fs

signal = a0 * np.sin(2*np.pi*f0*tt) + a1 * np.sin(2*np.pi*f1*tt)
mod_signal=np.abs(np.fft.fft(signal))*2/N
mod_signal = 20 *np.log10(mod_signal)

# grafico
fig = plt.figure("Señal bitonal " , constrained_layout=True)
plt.title("Señal bitonal separada 10df")
plt.plot(freq[0:int(N/2)],mod_signal[0:int(N/2)])
plt.xlabel("Frecuencia normalizada [f/fs]")
plt.ylabel("Magnitud [dB]")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()


#%% Señales ej 2b
a1 = a0 * 10**(-265/20)
signal2 = a0 * np.sin(2*np.pi*f0*tt) + a1 * np.sin(2*np.pi*f1*tt)
mod_signal2 = np.abs(np.fft.fft(signal2))*2/N
mod_signal2 = 20 *np.log10(mod_signal2)

# grafico
fig = plt.figure("Señal bitonal con a1 minimo " , constrained_layout=True)
plt.title("Señal bitonal separada 10df")
plt.plot(freq[0:int(N/2)],mod_signal2[0:int(N/2)])
plt.xlabel("Frecuencia normalizada [f/fs]")
plt.ylabel("Magnitud [dB]")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()



#%% Genero ruido de cuantificacion 
a1 = a0 * 10**(-90/20)
signal3 = a0 * np.sin(2*np.pi*f0*tt) + a1 * np.sin(2*np.pi*f1*tt)
# para que se vea mejor el ruido ya que es muy deterministico agregarle  ruido con distribuzión
#uniforme de -/- 5 LSB si lo hago despues del cuentificador de -5 a +5
n = 16 # bits de cuatizacion
LSB = a0/2**n
ruido = np.random.uniform(-5*LSB, 5*LSB, N)
signal3 = signal3 + ruido

qsignal3 = np.round(signal3*(2**n)/2) / ((2**n)/2)
    
e16 = signal3 - qsignal3

mod_signal3 = np.abs(fft(qsignal3))*2/N # FFT de la señal cuantificada
mod_signal3 = 20 *np.log10(mod_signal3 + np.finfo(float).eps)
 
#tt = np.linspace(0,(N-1)/fs, N)  
#fig = plt.figure("Señal bitonal cuantificada en tiempo ")
#plt.title("Señal bitonal separada 10df")
#plt.plot(tt[:20], qsignal3[:20])
#plt.xlabel("Tiempo")
#plt.ylabel("Amplitud")
#plt.axhline(0, color="black")
#plt.axvline(0, color="black")
#plt.grid()



fig = plt.figure("Señal bitonal cuantificada ")
plt.title("Señal bitonal separada 10df")
plt.plot(freq[0:int(N/2)],mod_signal3[0:int(N/2)])
plt.xlabel("Frecuencia normalizada [f/fs]")
plt.ylabel("Magnitud [dB]")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()


