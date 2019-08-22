#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 19:20:28 2019

@author: nico
"""

# Testbench_senoidal

import matplotlib.pyplot as plt
import numpy as np

from pdsmodulos.senoidal import seno 



N  = 1000 # muestras
fs = 1000 # Hz
Df = fs / N
a0 = 1 # Volts
p0 = 0 # radianes
f0 = 10   # Hz
w  = 2 * np.pi * f0




tt, signal = seno(fs, f0, N, a0, p0)
plt.figure("Funcion  senoidal")
plt.plot(tt, signal,color='blue',label='sin(wt)')
plt.xlabel('tiempo [segundos]')
plt.ylabel('amplitud [V] ')
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.title('Funcion senoidal')
plt.ion()
#plt.savefig("funciones.eps")
plt.legend(loc=1)