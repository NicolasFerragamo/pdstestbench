#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 20:42:24 2019

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

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

fig_sz_x = 14
fig_sz_y = 13
fig_dpi = 80 # dpi

fig_font_family = 'Ubuntu'
fig_font_size = 16



#%% cargo el archivo ECG_TP$.mat
# para listar las variables que hay en el archivo
#sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
ecg_one_lead = ecg_one_lead.flatten(1)
cant_muestras = len(ecg_one_lead)

#%% Defino la fs y el eje de tiempo
fs = 1000
tt = np.linspace(0, cant_muestras, cant_muestras)

#%% genero el filtro de mediana original
the_start = time()
median1 = sig.medfilt(ecg_one_lead, 201) #200 ms
median2 = sig.medfilt(median1, 601) #600 ms
the_end = time()
tiempodft = the_end - the_start
signal = ecg_one_lead - median2
del the_start, the_end

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

#%% Medicion de la frecuencia de corte del filtro de multirate (me quedo con ek 95% de la energia, para eso utilizo la funcion cumsum)
K = 30
L = cant_muestras/K
ff2,Swelch = sig.welch(median2,fs=fs,nperseg=L,window='bartlett')
Swelch2 = 10*np.log10(Swelch)

plt.figure("Estimación de la señal interpolante con el método de Welch")
plt.title(" Estimación de la señal interpolante con el método de Welch")
plt.plot(ff2,Swelch2)
plt.xlabel('frecuecnia  [Hz]')
plt.ylabel('Amplitud db')
plt.grid()
plt.show()

# calculo la frecuencia de corte con el 95% de la enrgia
energia=np.zeros((int(L/2)+1))
np.cumsum(Swelch, out=energia)
limfreq = energia < 0.95*energia[-1]
for ii in range(len(limfreq)) :
     if limfreq[ii] == False:
          freq = ii
          break
 
# calculo la cantidad de pasadas   
nyq_frec = fs / 2
cant_pasadas  = nyq_frec/freq
cant_pasadas = np.log2(cant_pasadas)  #porque cada pasada divide a la mitad
cant_pasadas = np.round(cant_pasadas)




#%% Genero la interpolante utiliziando la técnica multirate

the_start = time()

decimation = ecg_one_lead
for jj in range(cant_pasadas):
     decimation = sig.decimate(decimation, 2)
 

median1_dec = sig.medfilt(decimation, 3) #200 ms
median2_dec = sig.medfilt(median1_dec, 5) #600 ms

interpolation = median2_dec
for jj in range(cant_pasadas):
     interpolation = sig.resample(interpolation,2*len(interpolation))
signal_int = ecg_one_lead - interpolation[0:len(ecg_one_lead)]     

the_end = time()
tiempodft_dec = the_end - the_start
del the_start, the_end     

#%% comparo los dos métodos en tiempo y en error absoluto

tiempo = tiempodft / tiempodft_dec
error = median2 - interpolation[0:len(ecg_one_lead)]
error_cuadratico = (median2 - interpolation[0:len(ecg_one_lead)])**2
valor_medio_real = np.mean(median2)
valor_medio_interpolate_signal = np.mean(interpolation) 
sesgo = np.abs(valor_medio_real - valor_medio_interpolate_signal)

error_cuadratico_medio = np.mean(error_cuadratico)
error__medio = np.mean(error)
var_error = np.var(error)

plt.figure("ECG 2", constrained_layout=True)
plt.title("ECG 2")
plt.plot(tt, ecg_one_lead, label='ECG original')
plt.plot(tt, signal, label='ECG filtrada completa')
plt.plot(tt, signal_int, label = 'ECG filtrada con resampleo')
plt.xlabel('Muestras')
plt.ylabel("Amplitud ")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.legend()
plt.show()

plt.figure("Comapración de estimadores", constrained_layout=True)
plt.title("Comparación de estimadores")
plt.plot(tt, median2, label='est med original')
plt.plot(tt, interpolation[0:len(ecg_one_lead)], label='est med resampling')
plt.xlabel('Muestras')
plt.ylabel("Amplitud ")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.legend()
plt.show()

plt.figure("Error cuadrático de estimadores", constrained_layout=True)
plt.title("Error cuadrático de estimadores")
plt.plot(tt, error_cuadratico, label='error cuadrático')
plt.plot(tt, np.ones((len(ecg_one_lead)))*error_cuadratico_medio, label='media')
plt.xlabel('Muestras')
plt.ylabel("Amplitud ")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.legend()
plt.show()

plt.figure("Histograma de errores")
plt.hist(error, bins=50, alpha=1, edgecolor = 'black',  linewidth=1, label="error")
plt.legend(loc = 'upper right')
plt.ylabel('frecuencia')
plt.xlabel('valores')
plt.title('Histograma de errores' )
plt.show()

#%% Zoom regions

# Segmentos de interés
regs_interes = ( 
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )
for ii in regs_interes:
    
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
    #hace el clipeo para salvar a los indices otra forma es el modulo N (le sumas N para que ingece
    #por el otro extremo y queda circular en 'C' se hace x % 5 )
    plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi= fig_dpi, facecolor='w', edgecolor='k')
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', lw=2)
    plt.plot(zoom_region, interpolation[zoom_region], label='interpolante resamplig')
    plt.plot(zoom_region, median2[zoom_region], label='interpolante')
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
            
    plt.show()

#%% Presentación de resultados
tus_resultados_per = [ 
                   [ tiempodft,valor_medio_real, '-' , '-'], # <-- acá debería haber numeritos :)
                   [ tiempodft_dec, valor_medio_interpolate_signal, '-', '-'], # <-- acá debería haber numeritos :)
                 ]
df = pd.DataFrame(tus_resultados_per, columns=['$tiempo', '$media', 'media_error', 'varianza'],
               index=['interpolante real','interpolante resamplleada'])

print("\n")
print(df)
