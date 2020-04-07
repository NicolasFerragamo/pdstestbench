#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 23:39:01 2019

@author: nico
"""

import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt


N  = 16
fs = 1000
f0 = 50
tt = np.linspace(0,(N - 1)/fs, N)
w  = 2 * np.pi * f0
df = fs / N

signal = np.sin(w * tt) + np.sin(2*w * tt) + np.sin(3*w * tt) + np.sin(4*w * tt) 
plt.figure("Funcion  senoidal f0=fs/2 con po=pi/2")
plt.plot(tt, signal,color='blue',label='sin(wt)')
plt.xlabel('tiempo [segundos]')
plt.ylabel('amplitud [V] ')
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.title('Funcion senoidal')
plt.ion()
plt.legend(loc=1)

fftsignal = fft(signal)
mod_signal = np.abs(fftsignal)/ N
fase_signal = np.angle(fftsignal)

plt.figure()
plt.subplot(2,1,1)
freq = np.linspace(0, (N-1)*df, N) / fs 
freq = np.linspace(0, (N-1), N) 
plt.plot(freq, mod_signal, label='modulo ', marker='.', linestyle='None')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.grid()
plt.title('Modulo de la señal')


plt.subplot(2,1,2)
plt.plot(freq[:int(N/2)],  mod_signal[:int(N/2)], label='fase ', marker='.', linestyle='None')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel('Fse de la señal')
plt.grid()
plt.title('fase de la señal ')
plt.tight_layout() #para ajustar el tamaño de lo contrario se puperpinan los titulos
plt.show()

