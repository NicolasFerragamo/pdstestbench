#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:59:32 2019

@author: nico
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import timeit

from pdsmodulos.signals import signals as sg 
from pdsmodulos.signals import FFT

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

fs = 1000 # Hz
a0 = 1 # Volts
p0 = 0 # radianes
f0 = 100 # Hz
w  = 2 * np.pi * f0


N = np.array([16, 32, 64, 128, 256, 512, 1024, 2048])

signal = []
fftsignal= []
tus_resultados=np.zeros(8)


#%% Generacion de la señal senoidal

ii = 0
while ii < 8:
     tt,signal = sg.seno (fs, f0, N[ii], a0, p0)
     the_start = timeit.timeit()
     np.fft.fft(signal)
     the_end = timeit.timeit()
     tus_resultados[ii] = the_start - the_end
     ii += 1
     
#%% calculando el tiempo de mi DFT


