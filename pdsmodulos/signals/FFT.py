#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:56:13 2019

@author: nico
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import cmath 

#%% Lipieza de gráficos

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

#%%  TRABAJAMOS CON LA FFT

def plotFFT (fftsignal, fs, N, y1l='amplitud [UA] ', y2l='Fase [rad] ', p1t=' ', p2t=' ', tp="FFT de la señal", loc1='upper right', loc2='upper right') :
    
    mod_signal = np.abs(fftsignal)
    fase_signal = np.angle(fftsignal)
    df= fs / N
    
    
#%% Ploteo de la FFT
    plt.figure(tp)
    plt.subplot(2,1,1)
    freq = np.linspace(0, (N-1)*df,N-1)/fs
    plt.plot(freq[0:int(N/2+1)], mod_signal[0:int(N/2+1)], color='blue', label='modulo')
    plt.xlabel('frecuecnia [Hz]')
    plt.ylabel(y1l)
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")
    plt.grid()
    plt.title('Modulo de la señal '+p1t)
    plt.legend(loc = loc1)


    plt.subplot(2,1,2)
    plt.plot(freq[0:int(N/2+1)], fase_signal[0:int(N/2+1)],color='red',label='fase')
    plt.xlabel('frecuecnia [Hz]')
    plt.ylabel(y2l)
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")
    plt.grid()
    plt.title('fase de la señal '+p2t)
    plt.legend(loc = loc2)
    plt.tight_layout() #para ajustar el tamaño de lo contrario se puperpinan los titulos
    plt.show()
    
    return 0

def myDFT (signal) :
    
    N =len(signal)
    Signal = np.empty(N)
    Signal[:N-1] = np.nan
    W = [ ]
    W = np.zeros((N,N),dtype=complex)  # tengo que limpiar la memoria, el vector
    for k in range (0, N-1):
        for n in range (0, N-1):
            W[k][n] = cmath.exp(-1j * 2 * np.pi * k * n/N)
    Signal = np.dot(W,  signal) 
    return Signal