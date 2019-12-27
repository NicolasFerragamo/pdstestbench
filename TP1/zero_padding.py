#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 21:56:51 2019

@author: nico
"""

import os
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('/home/nico/Documentos/facultad/6to_nivel/pds/git/pdstestbench')
from pdsmodulos.signals import signals as sg
from pdsmodulos.signals import FFT

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

N  = 1000 # muestras
fs = 1000 # Hz
df = fs / N
a0 = 1 # Volts
p0 = 0 # radianes
f0 = fs /4 + 0.5

M = [N, 2*N, 5*N, 10*N, 20*N]
fd1 = np.array(['N', '2*N', '5*N', '10*N', '20*N'])

#%% Genero los vectores
signal = np.zeros(N)

fftsignal1= np.zeros(M[0])
fftsignal2= np.zeros(M[1])
fftsignal3= np.zeros(M[2])
fftsignal4= np.zeros(M[3])
fftsignal5= np.zeros(M[4])


signal2 = np.zeros(M[1])
signal3 = np.zeros(M[2])
signal4 = np.zeros(M[3])
signal5 = np.zeros(M[4])



#%% generacion y muestreo de senoidal
tt, signal = sg.seno(fs, f0, N, a0, p0)

#%% concateno los ceros
signal2 = np.concatenate((signal, signal2), axis=0) 
signal3 = np.concatenate((signal, signal3), axis=0) 
signal4 = np.concatenate((signal, signal4), axis=0) 
signal5 = np.concatenate((signal, signal5), axis=0) 

#%% realización del experimento
fftsignal = np.fft.fft(signal)
FFT.plotFFT(fftsignal,fs,M, tp= 'FFT', c=0, l=fd1[0], db='on', m='.')
fftsignal2 = np.fft.fft(signal2)
FFT.plotFFT(fftsignal2,fs,M, tp= 'FFT', c=1, l=fd1[1], db='on', m='.')
fftsignal3 = np.fft.fft(signal3)
FFT.plotFFT(fftsignal3,fs,M, tp= 'FFT', c=2, l=fd1[2], db='on', m='.')
fftsignal4 = np.fft.fft(signal4)
FFT.plotFFT(fftsignal4,fs,M, tp= 'FFT', c=3, l=fd1[3], db='on', m='.')
fftsignal5 = np.fft.fft(signal5)
FFT.plotFFT(fftsignal5,fs,M, tp= 'FFT', c=4, l=fd1[4], db='on', m='.')

