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

#os.system ("clear") # limpia la terminal de python
#plt.close("all")    #cierra todos los graficos 

#%%  TRABAJAMOS CON LA FFT

def plotFFT (fftsignal, fs, N, y1l='Amplitud Normlizada [db] ', y2l='Fase [rad] ', p1t=' ', p2t=' ', tp="FFT de la señal", loc1='upper right', loc2='upper right', c=0, l=' ') :
    mod_signal = np.abs(fftsignal) *2 / N
    mod_signal = 20 *np.log10(mod_signal)
    fase_signal = np.angle(fftsignal)
    df= fs / N
    col= ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    
#%% Ploteo de la FFT
    plt.figure(tp)
    plt.subplot(2,1,1)
    freq = np.linspace(0, (N-1)*df, N) / fs
    plt.plot(freq[0:int(N/2+1)], mod_signal[0:int(N/2+1)], col[c], label='modulo '+ l, linestyle='-')
    plt.xlabel('frecuecnia normalizada f/fs [Hz]')
    plt.ylabel(y1l)
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")
    plt.grid()
    plt.title('Modulo de la señal '+p1t)
    plt.legend(loc = loc1)


    plt.subplot(2,1,2)
    plt.plot(freq[0:int(N/2+1)], fase_signal[0:int(N/2+1)], col[c], label='fase '+ l, linestyle='-')
    plt.xlabel('frecuecnia normalizada f/fs [Hz]')
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