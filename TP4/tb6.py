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
#from scipy.fftpack import fft
import scipy.io as sio
#from time import time
import pandas as pd
#from scipy.interpolate import CubicSpline

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

sio.whosmat('ECG_Limpio.mat')
mat_struct1 = sio.loadmat('ECG_Limpio.mat')
ecg_limpio = mat_struct1['ECG_Limpio']
ecg_limpio = ecg_limpio.flatten(1)

#%% correlacionar es lo mismo que convolucionar con la señal invertida
#utilizo la clase.reverce() para invertir los índices de los patrones

# qrs_1 = qrs_1[::-1]
# hb_1  = hb_1[::-1]
# hb_2  = hb_2[::-1]

#otra forma es usar np.flip(qrs_1)

#%% obtengo la correlación 
# tengo que normalizar para poder graficar
signal = ecg_limpio
correlation_QRS = sig.correlate(qrs_1, signal)
correlation_LN = sig.correlate(hb_1, signal)
correlation_LV = sig.correlate(hb_2, signal)


#%% deteccion de los latidos

QRS_detections_pos = sig.argrelmax(correlation_QRS,order=350)
QRS_detections_pos = QRS_detections_pos[0]

signal_detections_pos = sig.argrelmax(signal, order=250)
signal_detections_pos = signal_detections_pos[0]
      
hb_1_detections_pos = sig.argrelmax(correlation_LN,order=350)
hb_1_detections_pos = hb_1_detections_pos[0]

hb_2_detections_pos = sig.argrelmax(correlation_LV,order=350)
hb_2_detections_pos = hb_2_detections_pos[0]

#%% Gráficos 
correlation_QRS = correlation_QRS / np.max(correlation_QRS)
signal = signal / np.max(signal)
correlation_LN = correlation_LN / np.max(correlation_LN)
correlation_LV = correlation_LV / np.max(correlation_LV)


plt.figure("ECG y latidos detectados", constrained_layout=True)
plt.title("ECG y latidos detectados")
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

# Regiones de interes

regs_interes = ( 
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        )

for ii in regs_interes:
    
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]),
                            dtype='uint')
    #hace el clipeo para salvar a los indices otra forma es el modulo N (le sumas N para que ingece
    #por el otro extremo y queda circular en 'C' se hace x % 5 )
    plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi= fig_dpi, facecolor='w',
                edgecolor='k')
    plt.plot(zoom_region, signal[zoom_region], label='ECG Limpio', lw=2)
    plt.plot(zoom_region, correlation_QRS[zoom_region], label='QRS_detection')
    plt.plot(zoom_region, correlation_LN[zoom_region], label='LN_detection')
    plt.plot(zoom_region, correlation_LV[zoom_region], label='LV_detection')
    plt.title('ECG filtering from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    #plt.axis([zoom_region[0], zoom_region[-1], np.min(signal), np.max(signal)])
    axes_hdl = plt.gca()
    axes_hdl.legend()
    #axes_hdl.set_yticks(())
    plt.grid()
            
    plt.show()


#%% Gráficos 
qrs3 = ecg_one_lead[hb_1_detections_pos]
qrs2 = ecg_one_lead[QRS_detections_pos[:-1]]
qrs1 = ecg_one_lead[signal_detections_pos]
qrs = ecg_one_lead[qrs_detections]

plt.figure("Deteccion de los latidos", constrained_layout=True)
plt.title("Deteccion de los latidos")
plt.plot(tt, ecg_one_lead, label='Señal de ECG ')
plt.plot(qrs_detections, qrs, label='qrs_detection ', linestyle='None', marker='x')
plt.plot(signal_detections_pos, qrs1, label='maximo ', linestyle='None', marker='x')
plt.plot(QRS_detections_pos[:-1], qrs2, label=' QRS correlation', linestyle='None', marker='x')
plt.plot(hb_1_detections_pos, qrs3, label='LN correlation', linestyle='None', marker='x')
plt.xlabel('Muestras')
plt.ylabel("Amplitud ")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.legend()
plt.show()


# Regiones de interes

regs_interes = ( 
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        np.array([5, 5.2]) *60*fs, # minutos a muestras
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
    plt.plot(signal_detections_pos, qrs1, label='Argrelmax_detection ', 
              linestyle='None', marker='x')
    plt.plot(qrs_detections, qrs, label='qrs_detection', linestyle='None', 
              marker='x')
    plt.plot(QRS_detections_pos[:-1], qrs2, label='QRS_detection', linestyle='None', 
              marker='x')
    plt.plot(hb_1_detections_pos, qrs3, label='LN_detection', linestyle='None', 
              marker='x')
    plt.title('ECG filtering from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    plt.axis([zoom_region[0], zoom_region[-1], np.min(ecg_one_lead),
              np.max(ecg_one_lead)])
    axes_hdl = plt.gca()
    axes_hdl.legend()
    #axes_hdl.set_yticks(())
    plt.grid()
            
    plt.show()


#%% Métrica de cuantificación.

"""
  Propongo utilizar la varianza de la diferencia entre las deteccipones dadas y 
 el algoritomo Pan-Tompkins , no es una buena medida porque hay detecciones que 
 solo estan con un algoritmo y no con el otro, por lo tanto me quedo solo con las
 detecciones correctas y no las 2 erroneas.
 
 1° métrica podria ser comparar la cantidad de detecciones 
 2° métrica podria ser calcular el error pero solo de las detecciones "correctas"
"""
# 1°
QRS_error_cant_detecciones = np.abs(cant_detections - len(QRS_detections_pos))
hb_1_error_cant_detecciones = np.abs(cant_detections - len(hb_1_detections_pos))
hb_2_error_cant_detecciones = np.abs(cant_detections - len(hb_2_detections_pos))
#2°
# la muestra 0 y la 305 es erronea

QRS_error1 = np.abs(QRS_detections_pos[1:305] - qrs_detections[0:304])
QRS_error2 = np.abs(QRS_detections_pos[306:] - qrs_detections[304:])
QRS_error_abs = np.concatenate((QRS_error1,QRS_error2),axis=0) 
QRS_error_rel = QRS_error_abs /qrs_detections
QRS_error_rel_porcentual = QRS_error_rel *100
QRS_Varianza = np.var(QRS_error_rel)
del QRS_error1, QRS_error2


hb_1_error1 = np.abs(hb_1_detections_pos[1:305] - qrs_detections[0:304])
hb_1_error2 = np.abs(hb_1_detections_pos[306:] - qrs_detections[304:])
hb_1_error_abs = np.concatenate((hb_1_error1,hb_1_error2),axis=0) 
hb_1_error_rel = hb_1_error_abs /qrs_detections
hb_1_error_rel_porcentual = hb_1_error_rel *100
hb_1_varianza = np.var(hb_1_error_rel)
del hb_1_error1, hb_1_error2


hb_2_error1 = np.abs(hb_2_detections_pos[1:305] - qrs_detections[0:304])
hb_2_error2 = np.abs(hb_2_detections_pos[306:] - qrs_detections[304:])
hb_2_error_abs = np.concatenate((hb_2_error1,hb_2_error2),axis=0) 
hb_2_error_rel = hb_2_error_abs /qrs_detections
hb_2_error_rel_porcentual = hb_2_error_rel *100
hb_2_varianza = np.var(hb_2_error_rel)
del hb_2_error1, hb_2_error2

#%% presentación de resultados

QRS_NF = 0
QRS_PF = 1
QRS_PV = 1
QRS_NV = 1
QRS_TA = (QRS_NF + QRS_PF) *100/QRS_PV

hb_1_NF = 0
hb_1_PF = 1
hb_1_PV = 1
hb_1_NV = 1
hb_1_TA = (hb_1_NF + hb_1_PF) *100/hb_1_PV

hb_2_NF = 0
hb_2_PF = 1
hb_2_PV = 1
hb_2_NV = 1
hb_2_TA = (hb_2_NF + hb_2_PF) *100/hb_2_PV

tus_resultados_per = [ 
                       [ QRS_NF, QRS_PF, QRS_PV, QRS_NV, QRS_TA], # <-- acá debería haber numeritos :)
                       [ hb_1_NF, hb_1_PF, hb_1_PV, hb_1_NV, hb_1_TA],
                       [ hb_2_NF, hb_2_PF, hb_2_PV, hb_2_NV, hb_2_TA]
                     ]
errores = [
            [max(QRS_error_rel_porcentual), QRS_Varianza ],
            [max(hb_1_error_rel_porcentual), hb_1_Varianza ],
            [max(hb_2_error_rel_porcentual), hb_2_Varianza ]
          ]

df = pd.DataFrame(tus_resultados_per, columns=['NF', 'PF', 'PV', 'NV', 'TA %'],
               index=['  Complejo de ondas QRS normal','Latido normal', ' Latido de origen ventricular'])

df1 = pd.DataFrame(errores, columns=['error relativo maximo %', 'varianza'],
               index=['  Complejo de ondas QRS normal','Latido normal', ' Latido de origen ventricular'])

print("\n")
print(df)

print("\n")
print(df1)