#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 19:20:28 2019

@author: nico
"""
import os
import matplotlib.pyplot as plt
import numpy as np


from pdsmodulos.signals import senoidal 

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

N  = 1000 # muestras
fs = 1000 # Hz
df= fs / N
a0 = 1 # Volts
p0 = 0 # radianes
f0 = 10   # Hz
w  = 2 * np.pi * f0


#%% generacion y muestreo de senoidal

tt, signal = senoidal.seno(fs, f0, N, a0, p0)
ax = plt.figure("Funcion  senoidal")
plt.plot(tt, signal,color='blue',label='sin(wt)')
plt.xlabel('tiempo [segundos]')
plt.ylabel('amplitud [V] ')
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.title('Funcion senoidal')
plt.legend(loc = 'upper right')
plt.show()

#%%  TRABAJAMOS CON LA FFT

fftsignal = np.fft.fft(signal)
mod_signal = np.abs(fftsignal)
fase_signal = np.angle(fftsignal)


plt.figure("FFT de la señal")
plt.subplot(2,1,1)
freq = np.linspace(0, (N-1)*df,N-1)/fs
plt.plot(freq, mod_signal, color='blue', label='modulo')
plt.xlabel('frecuecnia [Hz]')
plt.ylabel('amplitud [V] ')
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.title('Modulo de la señal senoidal')
plt.legend(loc = 'upper center')


plt.subplot(2,1,2)
plt.plot(freq, fase_signal,color='red',label='fase')
plt.xlabel('frecuecnia [Hz]')
plt.ylabel('Fase [rad] ')
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.title('fase de la señal senoidal')
plt.legend(loc = 'upper center')
plt.tight_layout() #para ajustar el tamaño de lo contrario se puperpinan los titulos
plt.show()