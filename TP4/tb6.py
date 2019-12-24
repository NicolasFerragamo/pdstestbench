#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 21:14:00 2019

@author: nico
"""

import os
import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import scipy.io as sio
from time import time
import pandas as pd
from scipy.interpolate import CubicSpline

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

fig_sz_x = 14
fig_sz_y = 13
fig_dpi = 80 # dpi

fig_font_family = 'Ubuntu'
fig_font_size = 16



#%% cargo el archivo ECG_TP$.mat
# para listar las variables que hay en el archivo
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
ecg_one_lead = ecg_one_lead.flatten(1)
cant_muestras = len(ecg_one_lead)
qrs_detections = mat_struct['qrs_detections']
qrs_detections = qrs_detections.flatten(1)
cant_detections = len(qrs_detections)
qrs_1 = mat_struct['qrs_pattern1']
qrs_1 = qrs_1.flatten(1)
hb_1 = mat_struct['heartbeat_pattern1']
hb_1 = hb_1.flatten(1)
hb_2 = mat_struct['heartbeat_pattern2']
hb_2 = hb_2.flatten(1)

fs = 1000
tt = np.linspace(0, cant_muestras, cant_muestras)

#utilizo la señal de ECG filtrada con el filtro de mediana
median1 = sig.medfilt(ecg_one_lead, 201) #200 ms
median2 = sig.medfilt(median1, 601) #600 ms
the_end = time()

signal = ecg_one_lead - median2

#%% correlacionar es lo mismo que convolucionar con la señal invertida
#utilizo la clase.reverce() para invertir los índices de los patrones

# qrs_1 = qrs_1[::-1]
# hb_1  = hb_1[::-1]
# hb_2  = hb_2[::-1]

#otra forma es usar np.flip(qrs_1)

#%% obtengo la correlación 
# tengo que normalizar para poder graficar

signal = signal / np.max(signal)
correlation_QRS = sig.correlate(qrs_1, signal)
correlation_QRS = correlation_QRS / np.max(correlation_QRS)
correlation_LN = sig.correlate(hb_1, signal)
correlation_LN = correlation_LN / np.max(correlation_LN)
correlation_LV = sig.correlate(hb_2, signal)
correlation_LV = correlation_LV / np.max(correlation_LV)

plt.figure("Deteccion de los latidos", constrained_layout=True)
plt.title("Deteccion de los latidos")
plt.plot(tt, signal, label='Señal de ECG filtrada')
plt.plot(tt, correlation_QRS[len(qrs_1)-1:], label='patron QRS')
plt.plot(tt, correlation_LN[len(hb_1)-1:], label='patron Lat Normal')
plt.plot(tt, correlation_LV[len(hb_2)-1:], label='patron Lat Ventricular')
plt.xlabel('Muestras')
plt.ylabel("Amplitud ")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.legend()
plt.show()
