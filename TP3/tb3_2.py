#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 18:53:29 2019

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
mu = 0    # media (mu)
var = 2 # varianza
K = np.array([2, 5, 10, 20, 50], dtype=np.int)

#%% generación de señales    
signal = np.vstack(np.transpose([ np.random.normal(0, np.sqrt(var), N)  for j in range(Nexp)]))

#%% Bartlett
Sbar0 = np.vstack(np.transpose([sp.barlett(signal[:,ii], K=K[0], ax=0) for ii in range(Nexp)]))
Sbar1 = np.vstack(np.transpose([sp.barlett(signal[:,ii], K=K[1], ax=0) for ii in range(Nexp)]))
Sbar2 = np.vstack(np.transpose([sp.barlett(signal[:,ii], K=K[2], ax=0) for ii in range(Nexp)]))
Sbar3 = np.vstack(np.transpose([sp.barlett(signal[:,ii], K=K[3], ax=0) for ii in range(Nexp)]))
Sbar4 = np.vstack(np.transpose([sp.barlett(signal[:,ii], K=K[4], ax=0) for ii in range(Nexp)]))

#%% Cálculo de la energía 
energia0 = np.sum(Sbar0, axis=0) / (N/K[0])
energia1 = np.sum(Sbar1, axis=0) / (N/K[1])
energia2 = np.sum(Sbar2, axis=0) / (N/K[2])
energia3 = np.sum(Sbar3, axis=0) / (N/K[3])
energia4 = np.sum(Sbar4, axis=0) / (N/K[4])

#%% Valor medio muestreal
valor_medio_muestreal0 = np.mean(Sbar0, axis=1) 
valor_medio_muestreal1 = np.mean(Sbar1, axis=1) 
valor_medio_muestreal2 = np.mean(Sbar2, axis=1) 
valor_medio_muestreal3 = np.mean(Sbar3, axis=1) 
valor_medio_muestreal4 = np.mean(Sbar4, axis=1) 

#%% valor medio
valor_medio0 = np.mean(valor_medio_muestreal0, axis=0) 
valor_medio1 = np.mean(valor_medio_muestreal1, axis=0) 
valor_medio2 = np.mean(valor_medio_muestreal2, axis=0) 
valor_medio3 = np.mean(valor_medio_muestreal3, axis=0) 
valor_medio4 = np.mean(valor_medio_muestreal4, axis=0) 

#%% valor muestreal
var_muestreal0 = np.var(Sbar0, axis=1) 
var_muestreal1 = np.var(Sbar1, axis=1) 
var_muestreal2 = np.var(Sbar2, axis=1) 
var_muestreal3 = np.var(Sbar3, axis=1) 
var_muestreal4 = np.var(Sbar4, axis=1) 

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

plt.figure("Periodogramas de ruido blanco con K= " + A[0], constrained_layout=True)
plt.title("Periodogramas de ruido blanco con K= " + A[0])
plt.plot(freq0, Sbar0, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.figure("Promedio de los Periodogramas de ruido blanco con K= " + A[0], constrained_layout=True)
plt.title(" Promedio de los Periodogramas de ruido blanco con K= " + A[0])
plt.plot(freq0, valor_medio_muestreal0, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(valor_medio_muestreal0)-0.3, max(valor_medio_muestreal0)+0.3)
plt.grid()


#%%  Grafico de los resultados de K=5

plt.figure("Periodogramas de ruido blanco con K= " + A[1], constrained_layout=True)
plt.title("Periodogramas de ruido blanco con K= " + A[1])
plt.plot(freq1, Sbar1, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.figure("Promedio de los Periodogramas de ruido blanco con K= " + A[1], constrained_layout=True)
plt.title(" Promedio de los Periodogramas de ruido blanco con K= " + A[1])
plt.plot(freq1, valor_medio_muestreal1, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(valor_medio_muestreal1)-0.3, max(valor_medio_muestreal1)+0.3)
plt.grid()



#%%  Grafico de los resultados de K=10

plt.figure("Periodogramas de ruido blanco con K= " + A[2], constrained_layout=True)
plt.title("Periodogramas de ruido blanco con K= " + A[2])
plt.plot(freq2, Sbar2, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.figure("Promedio de los Periodogramas de ruido blanco con K= " + A[2], constrained_layout=True)
plt.title(" Promedio de los Periodogramas de ruido blanco con K= " + A[2])
plt.plot(freq2, valor_medio_muestreal2, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(valor_medio_muestreal2)-0.3, max(valor_medio_muestreal2)+0.3)
plt.grid()


#%%  Grafico de los resultados de K=20

plt.figure("Periodogramas de ruido blanco con K= " + A[3], constrained_layout=True)
plt.title("Periodogramas de ruido blanco con K= " + A[3])
plt.plot(freq3, Sbar3, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.figure("Promedio de los Periodogramas de ruido blanco con K= " + A[3], constrained_layout=True)
plt.title(" Promedio de los Periodogramas de ruido blanco con K= " + A[3])
plt.plot(freq3, valor_medio_muestreal3, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(valor_medio_muestreal3)-0.3, max(valor_medio_muestreal3)+0.3)
plt.grid()


#%%  Grafico de los resultados de K=50

plt.figure("Periodogramas de ruido blanco con K= " + A[4], constrained_layout=True)
plt.title("Periodogramas de ruido blanco con K= " + A[4])
plt.plot(freq4, Sbar4, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.figure("Promedio de los Periodogramas de ruido blanco con K= " + A[4], constrained_layout=True)
plt.title(" Promedio de los Periodogramas de ruido blanco con K= " + A[4])
plt.plot(freq4, valor_medio_muestreal4, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(valor_medio_muestreal4)-0.3, max(valor_medio_muestreal4)+0.3)
plt.grid()

#%% Gráfico de la varianza
varianza = [varianza0, varianza1, varianza2, varianza3, varianza4]

plt.figure("Varianza", constrained_layout=True)
plt.title("Varianza ")
plt.plot(K, varianza, marker='.')
plt.xlabel('número de ventanas K')
plt.ylabel("Varianza")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(varianza)-0.3, max(varianza)+0.3)
plt.grid()

#%% tbla de resultados
tus_resultados_bartlett = [ 
                   [valor_medio0, varianza0], # <-- acá debería haber numeritos :)
                   [valor_medio1, varianza1], # <-- acá debería haber numeritos :)
                   [valor_medio2, varianza2], # <-- acá debería haber numeritos :)
                   [valor_medio3, varianza3], # <-- acá debería haber numeritos :)
                   [valor_medio4, varianza4], # <-- acá debería haber numeritos :)
                 ]
df = pd.DataFrame(tus_resultados_bartlett, columns=['$s_B$', '$v_B$'],
               index=K)

print(df)
