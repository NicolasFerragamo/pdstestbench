#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:06:53 2019

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
##################
# a.2) Senoidal #
#################
a0 = 1 # Volts
p0 = 0 # radianes
f0 = fs/2  # Hz

# Insertar aquí el código para generar y visualizar la señal
##############################################################

tt, signal = sg.seno(fs, f0, N, a0, p0)
plt.figure("Funcion  senoidal f0 = fs/2")
plt.plot(tt, signal,color='blue',label='sin(wt)')
plt.xlabel('tiempo [segundos]')
plt.ylabel('amplitud [UA] ')
plt.grid()
plt.title('Funcion senoidal')
plt.ion()
plt.legend(loc=1)


##################
# a.3) Senoidal #
#################

a0 = 1       # Volts
p0 = np.pi/2 # radianes
f0 = fs/2    # Hz

# Insertar aquí el código para generar y visualizar la señal
##############################################################

tt, signal = sg.seno(fs, f0, N, a0, p0)
plt.figure("Funcion  senoidal f0=fs/2 con po=pi/2")
plt.plot(tt, signal,color='blue',label='sin(wt)')
plt.xlabel('tiempo [segundos]')
plt.ylabel('amplitud [V] ')
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.title('Funcion senoidal')
plt.ion()
plt.legend(loc=1)