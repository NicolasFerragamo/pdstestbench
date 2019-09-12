#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:02:15 2019

@author: nico
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import statistics as stats

from pdsmodulos.signals import signals as sg
#from pdsmodulos.signals import FFT

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

N  = 1000 # muestras
fs = 1000 # Hz
df = fs / N
a0 = 1 # Volts
p0 = 0 # radianes
f0 = fs /4 
Nexp = 500

M = [N, 5*N, 10*N]

a = -0.5 * df
b = 0.5 * df

#%% Genero los vectores
signal = []
signal = np.zeros((N, Nexp))

f1 = []
f1 = np.zeros((Nexp))

padding5 = []
padding5 = np.zeros((M[1], Nexp))

padding10 = []
padding10 = np.zeros((M[2], Nexp))

auxpadding5 = []
auxpadding5 = np.zeros((M[1] - N, Nexp))

auxpadding10 = []
auxpadding10 = np.zeros((M[2] - N, Nexp))

#%% Genero los vectores para la fft

fftsignal = []
fftsignal = np.zeros((N, Nexp), dtype=complex)

fftpadding5 = []
fftpadding5 = np.zeros((M[1], Nexp), dtype=complex)

fftpadding10 = []
fftpadding10 = np.zeros((M[2], Nexp), dtype=complex)


mod_signal = []
mod_signal = np.zeros((N, Nexp))

mod_padding5 = []
mod_padding5 = np.zeros((M[1], Nexp))

mod_padding10 = []
mod_padding10 = np.zeros((M[2], Nexp))

#%% Genero los vectores para los estimadores

k = []
k = np.zeros((Nexp))

k5 = []
k5 = np.zeros((Nexp))

k10 = []
k10 = np.zeros((Nexp))

#%% generación de frecuencias aleatorias
fa = np.random.uniform(a, b, size = (Nexp, 1)) # genera aleatorios
plt.hist(fa, bins=50, alpha=1, edgecolor = 'black',  linewidth=1)
plt.ylabel('frequencia')
plt.xlabel('valores')
plt.title('Histograma Uniforme')
plt.show()

#%% generación de señales

for ii in range(0, Nexp) :
     f1[ii] = f0 + fa[ii]
     
for ii in range(0, Nexp) :
     tt, signal[:,ii] = sg.seno(fs, f1[ii], N, a0, p0)
     
#%% Creacion de los vectores con padding
     
for ii in range(0, Nexp)  :
     padding5[:,ii] = np.concatenate((signal[:,ii], auxpadding5[:,ii]), axis=0)
     padding10[:,ii] = np.concatenate((signal[:,ii], auxpadding10[:,ii]), axis=0)
     

#%% realizo las FFTs

for ii in range (0, Nexp) :
          fftsignal[:,ii] = np.fft.fft(signal[:,ii])
          fftpadding5[:,ii] = np.fft.fft(padding5[:,ii])
          fftpadding10[:,ii] = np.fft.fft(padding10[:,ii])
          
          
#%% Obtengo los módulos de cada señal

for ii in range(0, Nexp) :
     mod_signal[:,ii] = np.abs(fftsignal[:,ii]) *2 / N
     mod_padding5[:,ii] = np.abs(fftpadding5[:,ii]) *2 / M[1]
     mod_padding10[:,ii] = np.abs(fftpadding10[:,ii]) *2 / M[2]
               
          
#%% prueba de los estadisticos
# en primer lugar realizo la prueba de maximizar el modulo de cada señal sin padding y obtengo los índices de losmaximos para cada experimento
     
max_signal1 = []
max_signal1 = np.zeros((Nexp))

mod_signal1 = []
mod_signal1 = np.zeros((int(N/2), Nexp))

mod_signal1 = mod_signal[:int(N/2), :]

max_signal1 = np.amax(mod_signal1, axis=0)

ii = 0
while (ii < Nexp) : 
     jj = 0
     while jj < int(N/2) :
          if (max_signal1[ii] == mod_signal[jj,ii]) :
               k[ii] = jj
          jj += 1
     ii += 1
     
 #%% prueba de los estadisticos
# en primer lugar realizo la prueba de maximizar el modulo de cada señal con padding5 y obtengo los índices de losmaximos para cada experimento
     
max_padding5 = []
max_padding5 = np.zeros((Nexp))

mod_padding5_1 = []
mod_padding5_1 = np.zeros((int(M[1]/2), Nexp))

mod_padding5_1 = mod_padding5[:int(M[1]/2), :]

max_padding5_1 = np.amax(mod_padding5_1, axis=0)

ii = 0
while (ii < Nexp) : 
     jj = 0
     while jj < int(M[1]/2) :
          if (max_padding5_1[ii] == mod_padding5[jj,ii]) :
               k5[ii] = jj
          jj += 1
     ii += 1
     
     
 #%% prueba de los estadisticos
# en primer lugar realizo la prueba de maximizar el modulo de cada señal con padding10 y obtengo los índices de losmaximos para cada experimento
     
max_padding10 = []
max_padding10 = np.zeros((Nexp))

mod_padding10_1 = []
mod_padding10_1 = np.zeros((int(M[2]/2), Nexp))

mod_padding10_1 = mod_padding10[:int(M[2]/2), :]

max_padding10_1 = np.amax(mod_padding10_1, axis=0)

ii = 0
while (ii < Nexp) : 
     jj = 0
     while jj < int(M[2]/2) :
          if (max_padding10_1[ii] == mod_padding10[jj,ii]) :
               k10[ii] = jj
          jj += 1
     ii += 1
     
     
 #%% error en la frecuencia

error_signal = []
error_signal = np.zeros((Nexp))

error_padding5 = []
error_padding5 = np.zeros((Nexp))

error_padding10 = []
error_padding10 = np.zeros((Nexp))
 

error_signal    = (k - f1) / f1
error_padding5  = (k5 - f1 * 5) / (5 * f1)
error_padding10 = (k10 - f1 * 10) / (10 * f1)


#%% Cálculos estadisticos

mean_error_signal = stats.mean(error_signal)
print("Valor medio de error_signal: ",mean_error_signal,"\n")

var_error_signal = stats.pvariance(error_signal)
print("Varianza de error_signal: ",var_error_signal,"\n")

dstd_error_signal= stats.pstdev(error_signal)
print("Desviación estandar de error_signal: ",dstd_error_signal,"\n")


# con 5*N zero padding
mean_error_padding5 = stats.mean(error_padding5)
print("Valor medio de error_padding5: ",mean_error_padding5,"\n")

var_error_padding5 = stats.pvariance(error_padding5)
print("Varianza de error_padding5: ",var_error_padding5,"\n")

dstd_error_padding5= stats.pstdev(error_padding5)
print("Desviación estandar de error_padding5: ",dstd_error_padding5,"\n")


# con 10*N zero padding
mean_error_padding10 = stats.mean(error_padding10)
print("Valor medio de error_padding10: ",mean_error_padding10,"\n")

var_error_padding10 = stats.pvariance(error_padding10)
print("Varianza de error_padding10: ",var_error_padding10,"\n")

dstd_error_padding10= stats.pstdev(error_padding10)
print("Desviación estandar de error_padding10: ",dstd_error_padding10,"\n")



#%% Ploteo de los errores

#plt.figure("Grafico de errores")
#plt.hist(error_signal, bins=50, alpha=1, edgecolor = 'black',  linewidth=1, label="signal")
#plt.hist(error_padding5, bins=50, alpha=1, edgecolor = 'black',  linewidth=1, label="padding5")
#plt.hist(error_padding10, bins=50, alpha=1, edgecolor = 'black',  linewidth=1, label="padding10")
#plt.legend(loc = 'upper right')
#plt.ylabel('frequencia')
#plt.xlabel('valores')
#plt.title('Histograma de errores')
#plt.show()
