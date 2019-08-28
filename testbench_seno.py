#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 19:20:28 2019

@author: nico
"""
import os
import matplotlib.pyplot as plt
import numpy as np


from pdsmodulos.signals import signals as sg 
from pdsmodulos.signals import FFT

#os.system ("clear") # limpia la terminal de python
#plt.close("all")    #cierra todos los graficos 

N  = 1000 # muestras
fs = 1000.0 # Hz
df= fs / N
a0 = 1 # Volts
p0 = 0# radianes
f0 = 100   # Hz
w  = 2 * np.pi * f0
duty = 50
width = 100
varianza = 1
SNR = 10

#%% generacion y muestreo de senoidal

tt, signal = sg.square(fs, f0 ,N, a0, p0,duty)
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

fftsignal = np.fft.fft(signal)
FFT.plotFFT(fftsignal,fs,N, tp= 'numpy FFT')

#%% mi DFT
fftsignal2 = FFT.myDFT(signal)
FFT.plotFFT(fftsignal2, fs, N, tp='my DFT')

