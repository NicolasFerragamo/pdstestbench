#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:22:29 2019

@author: nico
"""
#%% importo los paquetes necesarios
import os
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('/home/nico/Documentos/facultad/6to_nivel/pds/git/pdstestbench')
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

M = [N, N/10, N,10*N]
fd1 = np.array(['N', 'N/10', 'N', '10N'])

signal = np.zeros((N))

fftsignal    = np.zeros(int(M[0]), complex)
fftpadding01 = np.zeros(int(M[1]) + N, complex)
fftpadding1  = np.zeros(int(M[2]) + N, complex)
fftpadding10 = np.zeros(int(M[3]) + N, complex)

signal_padding01 = np.zeros(int(M[1]) + N, float)
signal_padding1  = np.zeros(int(M[2]) + N, float)
signal_padding10 = np.zeros(int(M[3]) + N, float)

aux_padding01 = np.zeros(int(M[1]), float)
aux_padding1  = np.zeros(int(M[2]), float)
aux_padding10 = np.zeros(int(M[3]), float)


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
fftsignal    = np.fft.fft(signal)
fftpadding01 = np.fft.fft(signal_padding01)
fftpadding1  = np.fft.fft(signal_padding1)
fftpadding10 = np.fft.fft(signal_padding10)


#%% prueba de los estadisticos
# en primer lugar realizo la prueba de maximizar el modulo de cada señal sin padding y obtengo los índices de losmaximos para cada experimento

f_signal   = 0
mod_signal = np.zeros(int((M[1]+N)/2))
mod_signal = abs(fftsignal[:int((M[1]+N)/2)])*2/(M[1] + N)
max_signal = np.amax(mod_signal, axis=0)

for ii in range(0, int((M[1]+N)/2)) :
     if (max_signal == mod_signal[ii]) :
          f_signal = ii
          
          
f_padding01   = 0
mod_padding01 = np.zeros(int((M[1]+N)/2))
mod_padding01 = abs(fftpadding01[:int((M[1]+N)/2)])*2/(M[1] + N)
max_padding01 = np.amax(mod_padding01, axis=0)

for ii in range(0, int((M[1]+N)/2)) :
     if (max_padding01 == mod_padding01[ii]) :
          f_padding01 = ii *10/11



f_padding1   = 0
mod_padding1 = np.zeros(int((M[2]+N)/2))
mod_padding1 = abs(fftpadding1[:int((M[2]+N)/2)])*2/(M[2] + N)
max_padding1 = np.amax(mod_padding1, axis=0)

for ii in range(0, int((M[2]+N)/2)) :
     if (max_padding1 == mod_padding1[ii]) :
          f_padding1 = ii / 2
          
          
f_padding10   = 0
mod_padding10 = np.zeros(int((M[3]+N)/2))
mod_padding10 = abs(fftpadding10[:int((M[3]+N)/2)])*2/(M[3] + N)
max_padding10 = np.amax(mod_padding10, axis=0)

for ii in range(0, int((M[3]+N)/2)) :
     if (max_padding10 == mod_padding10[ii]) :
          f_padding10 = ii / 11     
          
error_signal    = abs((f0 - f_signal)/f0)
error_padding01 = abs((f0 - f_padding01)/f0)   
error_padding1  = abs((f0 - f_padding1)/f0)   
error_padding10 = abs((f0 - f_padding10)/f0)   