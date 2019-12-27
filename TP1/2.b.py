#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:59:32 2019

@author: nico
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from time import time
import sys
sys.path.append('/home/nico/Documentos/facultad/6to_nivel/pds/git/pdstestbench')
from pdsmodulos.signals import signals as sg 
from pdsmodulos.signals import FFT

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

fs = 1000 # Hz
a0 = 1 # Volts
p0 = 0 # radianes
f0 = 100 # Hz
w  = 2 * np.pi * f0


N = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])


tiempofft=np.zeros(10)
tiempodft=np.zeros(10)

#%% Generacion de la señal senoidal

ii = 0
while ii < 8:
     tt,signal = sg.seno (fs, f0, N[ii], a0, p0)
     the_start = time()
     np.fft.fft(signal)
     the_end = time()
     tiempofft[ii] = the_end - the_start
     ii += 1
 
#%% calculando el tiempo de mi DFT     
ii = 0
while ii < 8:
     tt,signal = sg.seno (fs, f0, N[ii], a0, p0)
     the_start = time()
     FFT.myDFT(signal)
     the_end = time()
     tiempodft[ii] = the_end - the_start
     ii += 1     
     


import pandas as pd
index = ['16', '32', '64', '128', '256', '512', '1024', '2048',
          '4096', '8192']

data = {'N':index, 'FFT': tiempofft, 'myDFT':tiempodft}

df = pd.DataFrame(data)
df.set_index('N', inplace = True)
# select two columns 
print(df)

plt.figure("Gráfico comparatvo de tiempo entre la FFT y la DFT")
plt.plot(N, tiempofft, '*r', label='tiempo de la FFT')
plt.plot(N, tiempodft, '*b', label='tiempo de myDFT')
plt.ylabel('tiempo [s]')
plt.xlabel('N° de muestras')
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.xlim((0,4096))
plt.grid()
plt.title('Gráfico comparatvo de tiempo entre la FFT y la DFT ')
plt.legend(loc = 'upper right')
