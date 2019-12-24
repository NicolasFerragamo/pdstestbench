#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:50:04 2019

@author: nico
"""

import os
import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
#from scipy.fftpack import fft
import scipy.io as sio
#from time import time
#import pandas as pd
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

sio.whosmat('ECG_Limpio.mat')
mat_struct1 = sio.loadmat('ECG_Limpio.mat')
ecg_limpio = mat_struct1['ECG_Limpio']
ecg_limpio = ecg_limpio.flatten(1)

fs = 1000
tt = np.linspace(0, cant_muestras, cant_muestras)

qrs = ecg_one_lead[qrs_detections]
ventana2 = qrs_detections-90
ventana = ventana2 -50
#%% 

plt.figure("ECG", constrained_layout=True)
plt.title("ECG")
plt.plot(tt, ecg_one_lead, label='ECG original')
plt.plot(qrs_detections, qrs, label='QRS_detection', linestyle='None', marker='x')
plt.plot(ventana, np.ones(len(qrs_detections)), label='ventana de muestreo', 
         linestyle='None', marker='x')
plt.plot(ventana2, np.ones(len(qrs_detections)), label='ventana de muestreo2', 
         linestyle='None', marker='x')
plt.xlabel('Muestras')
plt.ylabel("Amplitud ")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.legend()
plt.show()

#%% 
'''Para poder ver mejor la ventana de retroceso conbiene promediar todos los 
latidos para evidenciar la onda p y t. para ello con el qr_detector retroceder 
200ms y avanzar 350ms para superponerlos. combiene quedarme en la parte incial 
que esta limpia y en la parte final donde no haya pruebas de estres. También hacer
un promedio de realizaciones y trabajar con eso.
'''

latidos = np.vstack(np.transpose([(ecg_one_lead[qrs_detections[ii]-200:
                                              qrs_detections[ii]+350]) 
                                     for ii in range(122)]))
# utilizo 122 porque es el número de latidos que estan al principio del archivo 
# que no esta afectado por la interferente.
latido_promedio = np.mean(latidos, axis=1)


latidos_limpios = np.vstack(np.transpose([(ecg_limpio[qrs_detections[ii]-200:
                                              qrs_detections[ii]+350]) 
                                     for ii in range(cant_detections)]))
latido_limpio_promedio = np.mean(latidos_limpios, axis=1)
#%% Gráfico de latidos
ttl = np.linspace(0,549,550) 

plt.figure("Latidos limpios", constrained_layout=True)
plt.title("Latidos limpios")
plt.plot(ttl, latidos_limpios)
plt.xlabel('Muestras')
plt.ylabel("Amplitud ")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.show()

plt.figure("Latido limpio Promedio", constrained_layout=True)
plt.title("Latidos limpio Promedio")
plt.plot(ttl, latido_limpio_promedio)
plt.xlabel('Muestras')
plt.ylabel("Amplitud ")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.show()

plt.figure("Latidos ECG original", constrained_layout=True)
plt.title("Latidos ECG original")
plt.plot(ttl, latidos)
plt.xlabel('Muestras')
plt.ylabel("Amplitud ")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.show()

plt.figure("Latido Promedio ECG original", constrained_layout=True)
plt.title("Latidos Promedio ECG original")
plt.plot(ttl, latido_promedio)
plt.xlabel('Muestras')
plt.ylabel("Amplitud ")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.show()
#%% filtro 
estimation = np.vstack(np.transpose([np.mean(ecg_one_lead[ventana[ii]:ventana2[ii]]) 
                                     for ii in range(cant_detections)]))

tiempo = (ventana + ventana2)/2
cs = CubicSpline(tiempo, estimation)

tt = np.linspace(0, cant_muestras-1, cant_muestras)
plt.figure("Estimación de linea de base")
plt.title("Estimación de linea de base")
plt.plot(tiempo, estimation, '.', label='Est. de linea de base sin interpolar')
plt.plot(tt, cs(tt), label='Est. de linea de base con CubicSpline')
plt.grid()
plt.show()

#%% ECG Filtrado
interpolante = cs(tt)
ECG_Filtrado = ecg_one_lead - interpolante.flatten(1)

plt.figure("ECG Filtrado", constrained_layout=True)
plt.title("ECG Filtrado")
plt.plot(tt, ecg_one_lead, label='ECG original')
plt.plot(tt, ECG_Filtrado, label='ECG Filtrada est. linea de base')
plt.plot(tt, ecg_limpio, label='ECG Filtrada con filtro de mediana')
plt.xlabel('Muestras')
plt.ylabel("Amplitud ")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.legend()
plt.show()

#%% Regiones de interes

regs_interes = ( 
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )

for ii in regs_interes:
    
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]),
                            dtype='uint')
    #hace el clipeo para salvar a los indices otra forma es el modulo N (le sumas N para que ingece
    #por el otro extremo y queda circular en 'C' se hace x % 5 )
    plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi= fig_dpi, facecolor='w',
               edgecolor='k')
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', lw=2)
    plt.plot(zoom_region, ECG_Filtrado[zoom_region], 
             label='ECG Filtrada est. linea de base')
    plt.plot(zoom_region, ecg_limpio[zoom_region], 
             label='ECG Filtrada con filtro de mediana')
    plt.title('ECG filtering from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
            
    plt.show()