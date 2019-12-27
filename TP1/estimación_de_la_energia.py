#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 21:07:56 2019

@author: nico
"""
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('/home/nico/Documentos/facultad/6to_nivel/pds/git/pdstestbench')
from pdsmodulos.signals import signals as sg 
from pdsmodulos.signals import FFT

plt.close("all")    #cierra todos los graficos 

N  = 1000 # muestras
fs = 1000 # Hz
df = fs / N
a0 = 1 # Volts
p0 = 0 # radianes
f0 = fs /4
w  = 2 * np.pi * f0
duty = 50
width = 100
varianza = 1
SNR = 10

fd = np.array([0, 0.01, 0.25, 0.5],float)
fd1 = np.array(['f0', '0.01', '0.25', '0.5'])

signal = []
signal = np.zeros((N,4))
fftsignal= []
fftsignal= np.zeros((N,4))


tt, signal[:,0] = sg.seno(fs, f0 + fd[0], N, a0, p0)

energia = 0
for ii in range(0,N):
     energia += signal[ii,0]**2

energia = energia / N
print ("la energía termporal es: " ,energia)

fftsignal = np.fft.fft(signal[:,0])
mod_fftsignal = np.abs(fftsignal)

energia2 = 0
for ii in range(0,N):
     energia2 += mod_fftsignal[ii]**2
     
energia2 = energia2/N**2
print ("la energía por frecuencia es: " ,energia2)
     