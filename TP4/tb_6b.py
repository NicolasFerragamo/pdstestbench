#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 11:33:07 2019

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

fs = 1000
tt = np.linspace(0, cant_muestras, cant_muestras)


'''
 Algoritmo de Pan–Tompkins
      _________       _____________       ___________       ______________       ______________
     |        |      |            |      |          |      |             |      |             |
     | Filtro |      | Filtro     |      | Elevar   |      | Integracion |      | Detección   |
 --->| pasa   |----->| Derivativo |----->|    al    |----->| por ventana |----->| de puntos   |
     | Banda  |      |            |      | Cuadrado |      | deslizante  |      |   QRS       |
     |________|      |____________|      |__________|      |_____________|      |_____________|
'''
#%% Filtro pasa banda 
'''
      _________      
     |        | 
     | Filtro |      
 --->| pasa   |----->
     | Banda  |      
     |________|      
'''
nyq_frec = fs / 2
#utilizo técnicas multirate para poder implementar el filtro
nyq_frec = nyq_frec/8

# filter design
ripple = 0.1 # dB
atenuacion = 40. # dB

ws1 = 0.4# 0.21 #Hz podria ser 0.05 pero uso la media geométrica para que sea simétrico
wp1 = 0.5#0.3 #Hz
wp2 = 40 #Hz
ws2 = 50 #50 #Hz

frecs = np.array([0.0,         ws1,         wp1,     wp2,     ws2,         nyq_frec   ]) / nyq_frec
gains = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])
gains = 10**(gains/20)

bp_sos_cauer = sig.iirdesign(wp=np.array([wp1, wp2]) / nyq_frec, ws=np.array([ws1, ws2]) 
                             / nyq_frec, gpass=ripple, gstop=atenuacion, analog=False,
                             ftype='ellip', output='sos')
#bajo la frecuencia del ECG para que sea compatible
ecg_one_lead_decimate = sig.decimate(ecg_one_lead, 8)

# Procedemos al filtrado
ECG_f_cauer = sig.sosfiltfilt(bp_sos_cauer, ecg_one_lead_decimate)
ECG_f_cauer = sig.resample(ECG_f_cauer,8*len(ECG_f_cauer))

del ecg_one_lead_decimate

#%% Filtro derivativo

'''
      _____________      
     |            |      
     |  Filtro    |      
---->|            |----->
     | Derivativo |      
     |____________|      

'''

#ECG_Derivada = np.diff(ECG_f_cauer)

#Make impulse response
h = np.array([-1, -2, 0, 2, 1])/8
Delay = 2 # Delay in samples
#Apply filter
ECG_Derivada = np.convolve(ECG_f_cauer ,h);
ECG_Derivada = ECG_Derivada[Delay:cant_muestras+Delay]
ECG_Derivada = ECG_Derivada / np.max( np.abs(ECG_Derivada));
#%% Filtro derivativo

'''
      ____________      
     |           |      
     | Elevar    |      
---->|    al     |----->
     | cuadrado  |      
     |___________|      

'''
ECG_cuadrado = np.square(ECG_Derivada)
#%% Integración por ventana deslizante
'''
      ______________       
     |             |      
     | Integracion |      
---->| por ventana |----->
     | deslizante  |      
     |_____________|      

'''

#Moving Window Integration
#Make impulse response
h = np.ones((31))/31
Delay = 15 # Delay in samples

#Apply filter
ECG_Detection = sig.convolve(ECG_cuadrado,h)
ECG_Detection = ECG_Detection[Delay:cant_muestras+Delay]
ECG_Detection = ECG_Detection / np.max( np.abs(ECG_Detection));

#%% Detección de puntos QRS
'''
      _____________       
     |            |      
     | Detección  |      
---->| de puntos  |----->
     | QRS        |      
     |____________|      

'''

# Busco los máximos

QRS_detections_pos = sig.argrelmax(ECG_Detection,order=350)
QRS_detections_pos = QRS_detections_pos[0]



#%% Gráficos 
qrs1 = ecg_one_lead[QRS_detections_pos]
qrs = ecg_one_lead[qrs_detections]

plt.figure("Deteccion de los latidos", constrained_layout=True)
plt.title("Deteccion de los latidos")
plt.plot(tt, ecg_one_lead, label='Señal de ECG ')
plt.plot(QRS_detections_pos, qrs1, label='QRS_det Pan-Tompkins algorithm ', linestyle='None', marker='x')
plt.plot(qrs_detections, qrs, label='QRS_detection', linestyle='None', marker='x')
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
    plt.plot(QRS_detections_pos, qrs1, label='QRS_det Pan-Tompkins algorithm ', 
             linestyle='None', marker='x')
    plt.plot(qrs_detections, qrs, label='QRS_detection', linestyle='None', 
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
 solo estan con un algoritmo y no con el otro
 
 1° métrica podria ser comparar la cantidad de detecciones 
 2° métrica podria ser calcular el error pero solo de las detecciones "correctas"
"""
# 1°
error_cant_detecciones = np.abs(cant_detections - len(QRS_detections_pos))

#2°

error1 = np.abs(QRS_detections_pos[1:305] - qrs_detections[0:304])
error2 = np.abs(QRS_detections_pos[306:] - qrs_detections[304:])
error_abs = np.concatenate((error1,error2),axis=0) 
error_rel = error_abs /qrs_detections
error_rel_porcentual = error_rel *100

varianza = np.var(error_rel)

del error1, error2

#%% presentación de resultados

NF = 0
PF = 2
PV = 1903
NV = 1903
TA = (NF + PF) *100/PV
tus_resultados_per = [ 
                       [ NF, PF, PV, NV, TA] # <-- acá debería haber numeritos :)
                   
                     ]
errores = [
            [max(error_rel_porcentual), varianza ]
          ]

df = pd.DataFrame(tus_resultados_per, columns=['NF', 'PF', 'PV', 'NV', 'TA %'],
               index=[' Algoritmo de Pan–Tompkins'])

df1 = pd.DataFrame(errores, columns=['error relativo maximo %', 'varianza'],
               index=[' Algoritmo de Pan–Tompkins'])

print("\n")
print(df)

print("\n")
print(df1)