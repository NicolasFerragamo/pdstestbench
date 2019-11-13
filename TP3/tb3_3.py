#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:48:24 2019

@author: nico
"""

import os
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
from pdsmodulos.signals import spectral_estimation as sp
import pandas as pd

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

# Simular para los siguientes tamaños de señal
N = 1000
fs = 1000 # Hz
df = fs/N
Nexp = 200
over = 0.5
mu = 0    # media (mu)
var = 2 # varianza
K = np.array([2, 5, 10, 20, 50], dtype=np.int)
L = N/K
#%% generación de señales    
signal = np.vstack(np.transpose([ np.random.normal(0, np.sqrt(var), N)  for j in range(Nexp)]))

#%% Welch
Swelch0 = np.vstack(np.transpose([sp.welch(signal[:,ii], L=L[0], over=over, win="Bartlett", ax=0) for ii in range(Nexp)]))
Swelch1 = np.vstack(np.transpose([sp.welch(signal[:,ii], L=L[1], over=over, win="Bartlett", ax=0) for ii in range(Nexp)]))
Swelch2 = np.vstack(np.transpose([sp.welch(signal[:,ii], L=L[2], over=over, win="Bartlett", ax=0) for ii in range(Nexp)]))
Swelch3 = np.vstack(np.transpose([sp.welch(signal[:,ii], L=L[3], over=over, win="Bartlett", ax=0) for ii in range(Nexp)]))
Swelch4 = np.vstack(np.transpose([sp.welch(signal[:,ii], L=L[4], over=over, win="Bartlett", ax=0) for ii in range(Nexp)]))

#%% Cálculo de la energía 
energia0 = np.sum(Swelch0, axis=0) / (N/K[0])
energia1 = np.sum(Swelch1, axis=0) / (N/K[1])
energia2 = np.sum(Swelch2, axis=0) / (N/K[2])
energia3 = np.sum(Swelch3, axis=0) / (N/K[3])
energia4 = np.sum(Swelch4, axis=0) / (N/K[4])

#%% Valor medio muestreal
valor_medio_muestreal0 = np.mean(Swelch0, axis=1) 
valor_medio_muestreal1 = np.mean(Swelch1, axis=1) 
valor_medio_muestreal2 = np.mean(Swelch2, axis=1) 
valor_medio_muestreal3 = np.mean(Swelch3, axis=1) 
valor_medio_muestreal4 = np.mean(Swelch4, axis=1) 

#%% valor medio
valor_medio0 = np.mean(valor_medio_muestreal0, axis=0) 
valor_medio1 = np.mean(valor_medio_muestreal1, axis=0) 
valor_medio2 = np.mean(valor_medio_muestreal2, axis=0) 
valor_medio3 = np.mean(valor_medio_muestreal3, axis=0) 
valor_medio4 = np.mean(valor_medio_muestreal4, axis=0) 

#%% sesgo 
sesgo0 = np.abs(valor_medio0 - var)
sesgo1 = np.abs(valor_medio1 - var)
sesgo2 = np.abs(valor_medio2 - var)
sesgo3 = np.abs(valor_medio3 - var)
sesgo4 = np.abs(valor_medio4 - var)

#%% varianza muestreal
var_muestreal0 = np.var(Swelch0, axis=1) 
var_muestreal1 = np.var(Swelch1, axis=1) 
var_muestreal2 = np.var(Swelch2, axis=1) 
var_muestreal3 = np.var(Swelch3, axis=1) 
var_muestreal4 = np.var(Swelch4, axis=1) 

#%% Varianza
varianza0 =  np.mean(var_muestreal0, axis=0)
varianza1 =  np.mean(var_muestreal1, axis=0)
varianza2 =  np.mean(var_muestreal2, axis=0)
varianza3 =  np.mean(var_muestreal3, axis=0)
varianza4 =  np.mean(var_muestreal4, axis=0)


#%%  Grafico 
A = ["2", "5", "10", "20", "50"]

## ejes de tiempo
tt = np.linspace(0, (N-1)/fs, N)  

freq0 = np.linspace(0, (N-1)*df, int(N/K[0])) / fs
freq1 = np.linspace(0, (N-1)*df, int(N/K[1])) / fs
freq2 = np.linspace(0, (N-1)*df, int(N/K[2])) / fs
freq3 = np.linspace(0, (N-1)*df, int(N/K[3])) / fs
freq4 = np.linspace(0, (N-1)*df, int(N/K[4])) / fs




#%%  Grafico de los resultados de K=2

plt.figure("Estimador de Welch para ruido blanco con K= " + A[0], constrained_layout=True)
plt.subplot(1,2,1)
plt.title("Estimador de Welch con K= " + A[0])
plt.plot(freq0, Swelch0, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.subplot(1,2,2)
#plt.figure("Promedio de los Estimador de Welch para de ruido blanco con K= " + A[0], constrained_layout=True)
plt.title(" Promedio de Welch con K= " + A[0])
plt.plot(freq0, valor_medio_muestreal0, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(valor_medio_muestreal0)-0.01, max(valor_medio_muestreal0)+0.01)
plt.grid()
plt.tight_layout()

#%%  Grafico de los resultados de K=5

plt.figure("Estimador de Welch para ruido blanco con K= " + A[1], constrained_layout=True)
plt.subplot(1,2,1)
plt.title("Estimador de Welch con K= " + A[1])
plt.plot(freq1, Swelch1, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.subplot(1,2,2)
#plt.figure("Promedio de los Estimador de Welch para ruido blanco con K= " + A[1], constrained_layout=True)
plt.title(" Promedio de Welch con K= " + A[1])
plt.plot(freq1, valor_medio_muestreal1, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(valor_medio_muestreal1)-0.01, max(valor_medio_muestreal1)+0.01)
plt.grid()
plt.tight_layout()

#%%  Grafico de los resultados de K=10

plt.figure("Estimador de Welch para ruido blanco con K= " + A[2], constrained_layout=True)
plt.subplot(1,2,1)
plt.title("Estimador de Welch con K= " + A[2])
plt.plot(freq2, Swelch2, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.subplot(1,2,2)
#plt.figure("Promedio de los Estimador de Welch para de ruido blanco con K= " + A[2], constrained_layout=True)
plt.title(" Promedio de Welch con K= " + A[2])
plt.plot(freq2, valor_medio_muestreal2, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(valor_medio_muestreal2)-0.01, max(valor_medio_muestreal2)+0.01)
plt.grid()
plt.tight_layout()

#%%  Grafico de los resultados de K=20

plt.figure("Estimador de Welch para ruido blanco con K= " + A[3], constrained_layout=True)
plt.subplot(1,2,1)
plt.title("Estimador de Welch con K= " + A[3])
plt.plot(freq3, Swelch3, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.subplot(1,2,2)
#plt.figure("Promedio de los Estimador de Welch para ruido blanco con K= " + A[3], constrained_layout=True)
plt.title(" Promedio de Welch con K= " + A[3])
plt.plot(freq3, valor_medio_muestreal3, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(valor_medio_muestreal3)-0.01, max(valor_medio_muestreal3)+0.01)
plt.grid()
plt.tight_layout()

#%%  Grafico de los resultados de K=50
plt.figure("Estimador de Welch para ruido blanco con K= " + A[4], constrained_layout=True)
plt.subplot(1,2,1)
plt.title("Estimador de Welch con K= " + A[4])
plt.plot(freq4, Swelch4, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()
plt.subplot(1,2,2)
#plt.figure("Promedio de los Estimador de Welch para ruido blanco con K= " + A[4], constrained_layout=True)
plt.title(" Promedio de Welch con K= " + A[4])
plt.plot(freq4, valor_medio_muestreal4, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(valor_medio_muestreal4)-0.01, max(valor_medio_muestreal4)+0.01)
plt.grid()
plt.tight_layout()

#%% Gráfico de la varianza
varianza = [varianza0, varianza1, varianza2, varianza3, varianza4]
sesgo =[sesgo0, sesgo1, sesgo2, sesgo3, sesgo4]

plt.figure("Consistencia del estimador", constrained_layout=True)
plt.subplot(1,2,1)
plt.title("Sesgo")
plt.plot(K, sesgo, marker='.')
plt.xlabel('número de ventanas K')
plt.ylabel("sesgo")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(sesgo)-0.01, max(sesgo)+0.01)
plt.grid()

plt.subplot(1,2,2)
plt.title("Varianza ")
plt.plot(K, varianza, marker='.')
plt.xlabel('número de ventanas K')
plt.ylabel("Varianza")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(varianza)-0.01, max(varianza)+0.01)
plt.grid()
plt.tight_layout()

#%% tbla de resultados
tus_resultados_welch = [ 
                   [sesgo0, varianza0], # <-- acá debería haber numeritos :)
                   [sesgo1, varianza1], # <-- acá debería haber numeritos :)
                   [sesgo2, varianza2], # <-- acá debería haber numeritos :)
                   [sesgo3, varianza3], # <-- acá debería haber numeritos :)
                   [sesgo4, varianza4], # <-- acá debería haber numeritos :)
                 ]
df = pd.DataFrame(tus_resultados_welch , columns=['$s_B$', '$v_B$'],
               index=K)

print(df)
