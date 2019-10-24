#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 18:49:21 2019

@author: nico
"""

#%% importo los paquetes necesarios
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from pdsmodulos.signals import signals as sg 
from pdsmodulos.signals import FFT

#%% limpio el entorno
os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

#%% Estalesco los datos necesarios
N  = 1000 # muestras
fs = 1000 # Hz
df = fs / N
a0 = 1 # Volts
p0 = 0 # radianes
f0 = fs / 4 + 0.5*df# Hz

#%% Genero las variables necesarias

M = [N/10, N,10*N]
fd1 = np.array(['N/10', 'N', '10N'])

signal     = np.zeros((N))

fftpadding01 = np.zeros(int(M[0]) + N, complex)
fftpadding1  = np.zeros(int(M[1]) + N, complex)
fftpadding10 = np.zeros(int(M[2]) + N, complex)

signal_padding01 = np.zeros(int(M[0]) + N, float)
signal_padding1  = np.zeros(int(M[1]) + N, float)
signal_padding10 = np.zeros(int(M[2]) + N, float)

aux_padding01 = np.zeros(int(M[0]), float)
aux_padding1  = np.zeros(int(M[1]), float)
aux_padding10 = np.zeros(int(M[2]), float)
#%% generacion y muestreo de las senoidal

tt, signal = sg.seno(fs, f0, N, a0, p0)
  
#%% concateno los ceros
signal_padding01 = np.concatenate((signal, aux_padding01), axis=0) 
signal_padding1  = np.concatenate((signal, aux_padding1), axis=0) 
signal_padding10 = np.concatenate((signal, aux_padding10), axis=0) 

del aux_padding01
del aux_padding1
del aux_padding10

#%% Gŕaficos de las señales frecuenciales
fftpadding01 = fft(signal_padding01)
FFT.plotFFT(fftpadding01,fs, tp= 'FFT', c=0, l=fd1[0], db='on', m='.')

fftpadding1 = fft(signal_padding1)
FFT.plotFFT(fftpadding1 ,fs, tp= 'FFT', c=1, l=fd1[1], db='on', m='.')

fftpadding10 = fft(signal_padding10)
FFT.plotFFT(fftpadding10,fs, tp= 'FFT', c=2, l=fd1[2], db='on', m='.')

