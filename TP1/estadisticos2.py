#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:31:38 2019

@author: nico
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import statistics as stats

from pdsmodulos.signals import signals as sg
from pdsmodulos.signals import FFT

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

signal = np.zeros((N, Nexp))

f1 = np.zeros((Nexp))

padding5 = np.zeros((M[1], Nexp))
padding10 = np.zeros((M[2], Nexp))

auxpadding5 = np.zeros((M[1] - N, Nexp))
auxpadding10 = np.zeros((M[2] - N, Nexp))

#%% Genero los vectores para la fft

fftsignal = np.zeros((N, Nexp), dtype=complex)
fftpadding5 = np.zeros((M[1], Nexp), dtype=complex)
fftpadding10 = np.zeros((M[2], Nexp), dtype=complex)

mod_signal = np.zeros((N, Nexp))
mod_padding5 = np.zeros((M[1], Nexp))
mod_padding10 = np.zeros((M[2], Nexp))

#%% Genero los vectores para los estimadores

k = np.zeros((Nexp))
k5 = np.zeros((Nexp))
k10 = np.zeros((Nexp))

#%%  Genero los vectores para guardar los errores

error_signal = np.zeros((Nexp))
error_padding5 = np.zeros((Nexp))
error_padding10 = np.zeros((Nexp))  

#%%  Genero las señales para guardar las energias

energia_tiempo = np.zeros((Nexp))
energia_frecuencia = np.zeros((Nexp))
energia_padding5 = np.zeros((Nexp))
energia_padding10 = np.zeros((Nexp))

#%% generación de frecuencias aleatorias
fa = np.random.uniform(a, b, size = (Nexp)) # genera aleatorios

plt.hist(fa, bins=20, alpha=1, edgecolor = 'black',  linewidth=1)
plt.ylabel('frequencia')
plt.xlabel('valores')
plt.title('Histograma Uniforme')
plt.savefig("Histograma.png")
plt.show()

#%% generación de señales

f1 = f0 + fa

del fa     

tt = np.linspace(0, (N-1)/fs, N)     

signal = np.vstack(np.transpose([a0 * np.sin(2*np.pi*j*tt) for j in f1]))  

del tt   
#%% Creacion de los vectores con padding

padding5  = np.concatenate((signal, auxpadding5),  axis=0)  
padding10 = np.concatenate((signal, auxpadding10), axis=0)
#%% borro las variables auxiliares

del auxpadding5
del auxpadding10   
     
#%% realizo las FFTs

fftsignal    = np.fft.fft(signal,    axis=0)
fftpadding5  = np.fft.fft(padding5,  axis=0)
fftpadding10 = np.fft.fft(padding10, axis=0)         
          
#%% Obtengo los módulos de cada señal

for ii in range(0, Nexp) :
     mod_signal[:,ii]    = np.abs(fftsignal[:,ii])
     mod_padding5[:,ii]  = np.abs(fftpadding5[:,ii]) 
     mod_padding10[:,ii] = np.abs(fftpadding10[:,ii]) 
               
#%% grafico una señal con cero padding

FFT.plotFFT( fftsignal[:,0], fs, N, y1l='Amplitud [UA] ', y2l='Fase [rad] ',
              c=0, db='on', tipo='plot', m='.', col_ax='off',  l=' sin padding')  
  
FFT.plotFFT( fftpadding5[:,0], fs, N, y1l='Amplitud [UA] ', y2l='Fase [rad] ',
              c=1, db='on', tipo='plot', m='.', col_ax='off',  l=' padding5')    

FFT.plotFFT( fftpadding10[:,0], fs, N, y1l='Amplitud [UA] ', y2l='Fase [rad] ',
              c=2, db='on', tipo='plot', m='.', col_ax='off',  l=' padding10')   
      
#%% prueba de los estadisticos
# en primer lugar realizo la prueba de maximizar el modulo de cada señal sin padding y obtengo los índices de losmaximos para cada experimento
     
max_signal1 = np.zeros((Nexp))
mod_signal1 = np.zeros((int(N/2), Nexp))
mod_signal1 = mod_signal[:int(N/2), :]
max_signal1 = np.amax(mod_signal1, axis=0)
k = np.argmax(mod_signal1, axis=0)  

#%% prueba de los estadisticos
# en primer lugar realizo la prueba de maximizar el modulo de cada señal con padding5 y obtengo los índices de losmaximos para cada experimento
     
max_padding5 = np.zeros((Nexp))
mod_padding5_1 = np.zeros((int(M[1]/2), Nexp))
mod_padding5_1 = mod_padding5[:int(M[1]/2), :]
max_padding5 = np.amax(mod_padding5_1, axis=0)
k5 = np.argmax(mod_padding5_1, axis=0)     
     
#%% prueba de los estadisticos
# en primer lugar realizo la prueba de maximizar el modulo de cada señal con padding10 y obtengo los índices de los maximos para cada experimento
     
max_padding10 = np.zeros((Nexp))
mod_padding10_1 = np.zeros((int(M[2]/2), Nexp))
mod_padding10_1 = mod_padding10[:int(M[2]/2), :]
max_padding10 = np.amax(mod_padding10_1, axis=0)
k10 = np.argmax(mod_padding10_1, axis=0)
     
#%% Calculo la energia de las señales en tiempo (valor real)
energia_tiempo = np.sum(signal**2, axis=0) / N

#%% Estimación puntual 
     
#Calculo la energia de las señales en frecuencia (valor estimado)
energia_frecuencia = (max_signal1**2)*2/(N**2)

# Calculo la energia de las señales en padding5(valor estimado)
energia_padding5 = (max_padding5 **2)*2/(N**2)

# Calculo la energia de las señales en padding10(valor estimado)
energia_padding10 = (max_padding10 **2) *2/(N**2)


#%% Calculo de error

# error en la señal sin cero padding
error_signal = energia_tiempo - energia_frecuencia

# error en la señal con cero padding5
error_padding5 = energia_tiempo - energia_padding5
     
# error en la señal con cero padding10
error_padding10 = energia_tiempo - energia_padding10

#%% Ploteo de los errores al estimar la enrgia 

plt.figure("Histograma de errores al estimar la energia")
plt.hist(error_signal, bins=20, alpha=1, edgecolor = 'black',  linewidth=1, label="signal")
plt.hist(error_padding5, bins=20, alpha=1, edgecolor = 'black',  linewidth=1, label="padding5")
plt.hist(error_padding10, bins=20, alpha=1, edgecolor = 'black',  linewidth=1, label="padding10")
plt.legend(loc = 'upper right')
plt.ylabel('frecuencia')
plt.xlabel('valores')
plt.title('histograma de errores al estimar la energia')
plt.show()   
     
#%%  Grafico de los resultados
plt.figure("Gráfico de los errores relativos de estimación espectral")
plt.plot(f1, error_signal, '*r', label='error señal')
plt.plot(f1, error_padding5, '*b', label='error padding5')
plt.plot(f1, error_padding10, '*g', label='error padding10')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel('error relativo')
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.xlim((249.5,250.5))
plt.grid()
plt.title('Errores relativos de estimación espectral')
plt.legend(loc = 'upper right')

     
#%% error en la frecuencia

error_fsignal = np.zeros((Nexp))
error_fpadding5 = np.zeros((Nexp))
error_fpadding10 = np.zeros((Nexp))
 
error_fsignal    = abs(k - f1)
error_fpadding5  = abs(k5 / 5 - f1) 
error_fpadding10 = abs(k10 / 10 - f1) 

#%% Cálculos estadisticos

mean_error_fsignal = stats.mean(error_fsignal)
print("Valor medio de error_fsignal: ",mean_error_fsignal,"\n")

var_error_fsignal = stats.variance(error_fsignal)
print("Varianza de error_sfignal: ",var_error_fsignal,"\n")

dstd_error_fsignal= stats.stdev(error_fsignal)
print("Desviación estandar de error_fsignal: ",dstd_error_fsignal,"\n")


# con 5*N zero padding
mean_error_fpadding5 = stats.mean(error_fpadding5)
print("Valor medio de error_fpadding5: ",mean_error_fpadding5,"\n")

var_error_fpadding5 = stats.variance(error_fpadding5)
print("Varianza de error_fpadding5: ",var_error_fpadding5,"\n")

dstd_error_fpadding5= stats.stdev(error_fpadding5)
print("Desviación estandar de error_fpadding5: ",dstd_error_fpadding5,"\n")


# con 10*N zero padding
mean_error_fpadding10 = stats.mean(error_fpadding10)
print("Valor medio de error_fpadding10: ",mean_error_fpadding10,"\n")

var_error_fpadding10 = stats.variance(error_fpadding10)
print("Varianza de error_fpadding10: ",var_error_fpadding10,"\n")

dstd_error_fpadding10= stats.stdev(error_fpadding10)
print("Desviación estandar de error_fpadding10: ",dstd_error_fpadding10,"\n")


#%% Ploteo de los errores

plt.figure("Histograma de errores al estimar la frecuencia")
plt.hist(error_fsignal, bins=20, alpha=1, edgecolor = 'black',  linewidth=1, label="signal")
plt.hist(error_fpadding5, bins=20, alpha=1, edgecolor = 'black',  linewidth=1, label="padding5")
plt.hist(error_fpadding10, bins=20, alpha=1, edgecolor = 'black',  linewidth=1, label="padding10")
plt.legend(loc = 'upper right')
plt.ylabel('frecuencia')
plt.xlabel('valores')
plt.title('histograma de errores al estimar la frecuencia')
plt.show()


#%% Ploteo de los errores
plt.figure("Grafico de errores al estimar la frecuencia")
plt.plot(f1, error_fsignal, '*r', label='error fseñal')
plt.plot(f1, error_fpadding5, '*b', label='error fpadding5')
plt.plot(f1, error_fpadding10, '*g', label='error fpadding10')
plt.legend(loc = 'upper right')
plt.grid()
plt.ylabel('error relativo')
plt.xlabel('frecuencias')
plt.title("Grafico de errores al estimar la frecuencia")
plt.show()
