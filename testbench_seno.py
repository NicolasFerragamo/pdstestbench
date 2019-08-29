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

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

N  = 1000 # muestras
fs = 1000 # Hz
df = fs / N
a0 = 1 # Volts
p0 = 0 # radianes
f0 = fs / 4 # Hz
w  = 2 * np.pi * f0
duty = 50
width = 100
varianza = 1
SNR = 10

fd = np.array([0.01, 0.25, 0.5],float)
fd1 = np.array(['0.01', '0.25', '0.5'])

signal = []
signal = np.zeros((N,3))
fftsignal= []
fftsignal= np.zeros((N,3))
#%% generacion y muestreo de senoidal
for ii in range(0,3):
    tt, signal[:,ii] = sg.seno(fs, f0 + fd[ii], N, a0, p0)
    
#tt, signal = sg.seno(fs, f0 + fd[2], N, a0, p0)
#ax = plt.figure("Funcion  senoidal")
#plt.plot(tt, signal,color='blue',label='sin(wt)')
#plt.xlabel('tiempo [segundos]')
#plt.ylabel('Amplitud [UA] ')
#plt.axhline(0, color="black")
#plt.axvline(0, color="black")
#plt.grid()
#plt.title('Funcion senoidal')
#plt.legend(loc = 'upper right')
#plt.show()

ii = 0
while ii < 3 : 
    fftsignal[:,ii] = np.fft.fft(signal[:,ii])
    FFT.plotFFT(fftsignal[:,ii],fs,N, tp= 'FFT', c=ii, l=fd1[ii])
    ii += 1
#
#fftsignal = np.fft.fft(signal)
#FFT.plotFFT(fftsignal,fs,N, tp= 'FFT')
#%% mi DFT
#fftsignal2 = FFT.myDFT(signal)
#FFT.plotFFT(fftsignal2, fs, N, tp='my DFT')

