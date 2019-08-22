#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 19:20:28 2019

@author: nico
"""

# Testbench_senoidal

import matplotlib.pyplot as plt
import numpy as np

from pdsmodulos.senoidal import seno 
import random


N  = 1000 # muestras
fs = 1000 # Hz
df= fs / N
a0 = 1 # Volts
p0 = 0 # radianes
f0 = 10   # Hz
w  = 2 * np.pi * f0




tt, signal = seno(fs, f0, N, a0, p0)
#plt.figure("Funcion  senoidal")
#plt.plot(tt, signal,color='blue',label='sin(wt)')
#plt.xlabel('tiempo [segundos]')
#plt.ylabel('amplitud [V] ')
#plt.axhline(0, color="black")
#plt.axvline(0, color="black")
#plt.grid()
#plt.title('Funcion senoidal')
#plt.ion()
#plt.savefig("funciones.eps")
#plt.legend(loc=1)

#%%  TRABAJAMOS CON LA FFT

fftsignal = np.fft.fft(signal)
mod_signal = np.abs(fftsignal)
fase_signal = np.angle(fftsignal)


plt.subplot(2,1,1)
plt.figure("Modulo de la señal senoidal")
freq = np.linspace(0, (N-1)*df,N-1)/fs
plt.plot(freq, mod_signal, color='blue', label='modulo')
plt.xlabel('frecuecnia [Hz]')
plt.ylabel('amplitud [V] ')
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.title('Modulo de la señal senoidal')
plt.legend(loc=1)



plt.subplot(2,1,2)
plt.plot(freq, fase_signal,color='red',label='fase')
plt.xlabel('frecuecnia [Hz]')
plt.ylabel('Fase [rad] ')
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.title('fase de la señal senoidal')
plt.legend(loc=1)