#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:41:32 2019

@author: nico
"""

import os
import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import scipy.io as sio

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

fig_sz_x = 14
fig_sz_y = 13
fig_dpi = 80 # dpi

fig_font_family = 'Ubuntu'
fig_font_size = 16


#Diseño de los filtros digitales

# para listar las variables que hay en el archivo
#sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
ecg_one_lead = ecg_one_lead.flatten(1)
cant_muestras = len(ecg_one_lead)
fs = 1000
tt = np.linspace(0, cant_muestras, cant_muestras)

median1 = sig.medfilt(ecg_one_lead, 201)
median2 = sig.medfilt(median1,601)

signal = ecg_one_lead - median2

plt.figure("ECG", constrained_layout=True)
plt.title("ECG")
plt.plot(tt, ecg_one_lead, label='ECG original')
plt.plot(tt, signal, label='ECG filtrada')
plt.xlabel('Muestras')
plt.ylabel("Amplitud ")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.legend()
plt.show()

