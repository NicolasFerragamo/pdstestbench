#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:21:51 2019

@author: nico
"""
import sys
sys.path.append('/home/nico/Documentos/facultad/6to_nivel/pds/git/pdstestbench')
import os
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
from pdsmodulos.signals import spectral_estimation as sp
import pandas as pd

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

# Simular para los siguientes tamaños de señal
N = np.array([10, 50, 100, 250, 500, 1000, 5000], dtype=np.int)
fs = 1000 # Hz
Nexp = 200
mu = 0    # media (mu)
var = 2 # varianza

#%% generación de señales    
signal0 = np.vstack(np.transpose([ np.random.normal(0, np.sqrt(var), N[0])  for j in range(Nexp)]))
signal1 = np.vstack(np.transpose([ np.random.normal(0, np.sqrt(var), N[1])  for j in range(Nexp)]))
signal2 = np.vstack(np.transpose([ np.random.normal(0, np.sqrt(var), N[2])  for j in range(Nexp)]))
signal3 = np.vstack(np.transpose([ np.random.normal(0, np.sqrt(var), N[3])  for j in range(Nexp)]))
signal4 = np.vstack(np.transpose([ np.random.normal(0, np.sqrt(var), N[4])  for j in range(Nexp)]))
signal5 = np.vstack(np.transpose([ np.random.normal(0, np.sqrt(var), N[5])  for j in range(Nexp)]))
signal6 = np.vstack(np.transpose([ np.random.normal(0, np.sqrt(var), N[6])  for j in range(Nexp)]))


#%% Periodograma
Sper0  = sp.periodogram(signal0, exp=Nexp, ax=0)
Sper1  = sp.periodogram(signal1, exp=Nexp, ax=0)
Sper2  = sp.periodogram(signal2, exp=Nexp, ax=0)
Sper3  = sp.periodogram(signal3, exp=Nexp, ax=0)
Sper4  = sp.periodogram(signal4, exp=Nexp, ax=0)
Sper5  = sp.periodogram(signal5, exp=Nexp, ax=0)
Sper6  = sp.periodogram(signal6, exp=Nexp, ax=0)

#%% Cálculo de la energía 
energia0 = np.sum(Sper0, axis=0) / N[0]
energia1 = np.sum(Sper1, axis=0) / N[1]
energia2 = np.sum(Sper2, axis=0) / N[2]
energia3 = np.sum(Sper3, axis=0) / N[3]
energia4 = np.sum(Sper4, axis=0) / N[4]
energia5 = np.sum(Sper5, axis=0) / N[5]
energia6 = np.sum(Sper6, axis=0) / N[6]

#%% Valor medio muestreal
valor_medio_muestreal0 = np.mean(Sper0, axis=1) 
valor_medio_muestreal1 = np.mean(Sper1, axis=1) 
valor_medio_muestreal2 = np.mean(Sper2, axis=1) 
valor_medio_muestreal3 = np.mean(Sper3, axis=1) 
valor_medio_muestreal4 = np.mean(Sper4, axis=1) 
valor_medio_muestreal5 = np.mean(Sper5, axis=1) 
valor_medio_muestreal6 = np.mean(Sper6, axis=1) 

#%% valor medio
valor_medio0 = np.mean(valor_medio_muestreal0, axis=0) 
valor_medio1 = np.mean(valor_medio_muestreal1, axis=0) 
valor_medio2 = np.mean(valor_medio_muestreal2, axis=0) 
valor_medio3 = np.mean(valor_medio_muestreal3, axis=0) 
valor_medio4 = np.mean(valor_medio_muestreal4, axis=0) 
valor_medio5 = np.mean(valor_medio_muestreal5, axis=0) 
valor_medio6 = np.mean(valor_medio_muestreal6, axis=0) 

#%% sesgo 
sesgo0 = np.abs(valor_medio0 - var)
sesgo1 = np.abs(valor_medio1 - var)
sesgo2 = np.abs(valor_medio2 - var)
sesgo3 = np.abs(valor_medio3 - var)
sesgo4 = np.abs(valor_medio4 - var)
sesgo5 = np.abs(valor_medio5 - var)
sesgo6 = np.abs(valor_medio6 - var)

#%% valor muestreal
var_muestreal0 = np.var(Sper0, axis=1) 
var_muestreal1 = np.var(Sper1, axis=1) 
var_muestreal2 = np.var(Sper2, axis=1) 
var_muestreal3 = np.var(Sper3, axis=1) 
var_muestreal4 = np.var(Sper4, axis=1) 
var_muestreal5 = np.var(Sper5, axis=1) 
var_muestreal6 = np.var(Sper6, axis=1) 

#%% Varianza
varianza0 =  np.mean(var_muestreal0, axis=0)
varianza1 =  np.mean(var_muestreal1, axis=0)
varianza2 =  np.mean(var_muestreal2, axis=0)
varianza3 =  np.mean(var_muestreal3, axis=0)
varianza4 =  np.mean(var_muestreal4, axis=0)
varianza5 =  np.mean(var_muestreal5, axis=0)
varianza6 =  np.mean(var_muestreal6, axis=0)


#%%  Grafico 
A = ["10", "50", "100", "250", "500", "1000", "5000"]
## ejes de tiempo
tt0 = np.linspace(0, (N[0]-1)/fs, N[0])  
tt1 = np.linspace(0, (N[1]-1)/fs, N[1]) 
tt2 = np.linspace(0, (N[2]-1)/fs, N[2])
tt3 = np.linspace(0, (N[3]-1)/fs, N[3]) 
tt4 = np.linspace(0, (N[4]-1)/fs, N[4]) 
tt5 = np.linspace(0, (N[5]-1)/fs, N[5]) 
tt6 = np.linspace(0, (N[6]-1)/fs, N[6]) 

ff0 = np.linspace(0,(N[0]-1)*fs/N[0], N[0])/fs
ff1 = np.linspace(0,(N[1]-1)*fs/N[1], N[1])/fs
ff2 = np.linspace(0,(N[2]-1)*fs/N[2], N[2])/fs
ff3 = np.linspace(0,(N[3]-1)*fs/N[3], N[3])/fs
ff4 = np.linspace(0,(N[4]-1)*fs/N[4], N[4])/fs
ff5 = np.linspace(0,(N[5]-1)*fs/N[5], N[5])/fs
ff6 = np.linspace(0,(N[6]-1)*fs/N[6], N[6])/fs


#%%  Grafico de los resultados de N=10
plt.figure("Gráfico de realizaciones de ruido blanco con N= " + A[0], constrained_layout=True)
plt.title("Gráfico de realizaciones de ruido blanco con N= " + A[0])
plt.plot(tt0,signal0)
plt.xlabel("tiempo [S]")
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.figure("Periodogramas de ruido blanco con N= " + A[0], constrained_layout=True)
plt.subplot(1,2,1)
plt.title("Estimador Periodograma con N= " + A[0])
plt.plot(ff0, Sper0, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.subplot(1,2,2)
#plt.figure("Promedio de los Periodogramas de ruido blanco con n°= " + A[0], constrained_layout=True)
plt.title("Promedio del Periodogramas con N= " + A[0])
plt.plot(ff0, valor_medio_muestreal0, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(valor_medio_muestreal0)-0.01, max(valor_medio_muestreal0)+0.01)
plt.grid()


#%%  Grafico de los resultados de N=50
plt.figure("Gráfico de realizaciones de ruido blanco con N= " + A[1], constrained_layout=True)
plt.title("Gráfico de realizaciones de ruido blanco con N= " + A[1])
plt.plot(tt1,signal1)
plt.xlabel("tiempo [S]")
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.figure("Periodogramas de ruido blanco con N= " + A[1], constrained_layout=True)
plt.subplot(1,2,1)
plt.title("Estimador Periodograma con N= " + A[1])
plt.plot(ff1, Sper1, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.subplot(1,2,2)
#plt.figure("Promedio de los Periodogramas de ruido blanco con n°= " + A[1], constrained_layout=True)
plt.title("Promedio del Periodogramas con N= " + A[1])
plt.plot(ff1, valor_medio_muestreal1, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(valor_medio_muestreal1)-0.01, max(valor_medio_muestreal1)+0.01)
plt.grid()
plt.tight_layout()


#%%  Grafico de los resultados de N=100
plt.figure("Gráfico de realizaciones de ruido blanco con N= " + A[2], constrained_layout=True)
plt.title("Gráfico de realizaciones de ruido blanco con N= " + A[2])
plt.plot(tt2,signal2)
plt.xlabel("tiempo [S]")
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.figure("Periodogramas de ruido blanco con N= " + A[2], constrained_layout=True)
plt.subplot(1,2,1)
plt.title("Estimador Periodograma con N= " + A[2])
plt.plot(ff2, Sper2, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.subplot(1,2,2)
#plt.figure("Promedio de los Periodogramas de ruido blanco con n°= " + A[2], constrained_layout=True)
plt.title("Promedio del Periodogramas con N= " + A[2])
plt.plot(ff2, valor_medio_muestreal2, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(valor_medio_muestreal2)-0.01, max(valor_medio_muestreal2)+0.01)
plt.grid()
plt.tight_layout()

#%%  Grafico de los resultados de N=250
plt.figure("Gráfico de realizaciones de ruido blanco con N= " + A[3], constrained_layout=True)
plt.title("Gráfico de realizaciones de ruido blanco con N= " + A[3])
plt.plot(tt3,signal3)
plt.xlabel("tiempo [S]")
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.figure("Periodogramas de ruido blanco con N= " + A[3], constrained_layout=True)
plt.subplot(1,2,1)
plt.title("Estimador Periodograma con N= " + A[3])
plt.plot(ff3, Sper3, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.subplot(1,2,2)
#plt.figure("Promedio de los Periodogramas de ruido blanco con n°= " + A[3], constrained_layout=True)
plt.title(" Promedio del Periodogramas con N= " + A[3])
plt.plot(ff3, valor_medio_muestreal3, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(valor_medio_muestreal3)-0.01, max(valor_medio_muestreal3)+0.01)
plt.grid()
plt.tight_layout()

#%%  Grafico de los resultados de N=500
plt.figure("Gráfico de realizaciones de ruido blanco con N= " + A[4], constrained_layout=True)
plt.title("Gráfico de realizaciones de ruido blanco con N= " + A[4])
plt.plot(tt4,signal4)
plt.xlabel("tiempo [S]")
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.figure("Periodogramas de ruido blanco con N= " + A[4], constrained_layout=True)
plt.subplot(1,2,1)
plt.title("Estimador Periodograma con N= " + A[4])
plt.plot(ff4, Sper4, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.subplot(1,2,2)
#plt.figure("Promedio de los Periodogramas de ruido blanco con n°= " + A[4], constrained_layout=True)
plt.title(" Promedio del Periodogramas con N= " + A[4])
plt.plot(ff4, valor_medio_muestreal4, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(valor_medio_muestreal4)-0.01, max(valor_medio_muestreal4)+0.01)
plt.grid()
plt.tight_layout()

#%%  Grafico de los resultados de N=1000
plt.figure("Gráfico de realizaciones de ruido blanco con N= " + A[5], constrained_layout=True)
plt.title("Gráfico de realizaciones de ruido blanco con N= " + A[5])
plt.plot(tt5,signal5)
plt.xlabel("tiempo [S]")
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.figure("Periodogramas de ruido blanco con N= " + A[5], constrained_layout=True)
plt.subplot(1,2,1)
plt.title("Estimador Periodograma con N= " + A[5])
plt.plot(ff5, Sper5, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.subplot(1,2,2)
#plt.figure("Promedio de los Periodogramas de ruido blanco con n°= " + A[5], constrained_layout=True)
plt.title(" Promedio del Periodogramas con N= " + A[5])
plt.plot(ff5, valor_medio_muestreal5, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(valor_medio_muestreal5)-0.01, max(valor_medio_muestreal5)+0.01)
plt.grid()
plt.tight_layout()


#%%  Grafico de los resultados de N=5000
plt.figure("Gráfico de realizaciones de ruido blanco con N= " + A[6], constrained_layout=True)
plt.title("Gráfico de realizaciones de ruido blanco con N= " + A[6])
plt.plot(tt6,signal6)
plt.xlabel("tiempo [S]")
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.figure("Periodogramas de ruido blanco con N= " + A[6], constrained_layout=True)
plt.subplot(1,2,1)
plt.title("Estimador Periodograma con N= " + A[6])
plt.plot(ff6, Sper6, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.grid()

plt.subplot(1,2,2)
#plt.figure("Promedio de los Periodogramas de ruido blanco con n°= " + A[6], constrained_layout=True)
plt.title(" Promedio del Periodogramas con N= " + A[6])
plt.plot(ff6, valor_medio_muestreal6, marker='.')
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Amplitud")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(valor_medio_muestreal6)-0.01, max(valor_medio_muestreal6)+0.01)
plt.grid()
plt.tight_layout()

#%% Gráfico de la varianza
varianza = [varianza0, varianza1, varianza2, varianza3, varianza4, varianza5, varianza6]
sesgo =[sesgo0, sesgo1, sesgo2, sesgo3, sesgo4, sesgo5, sesgo6]

plt.figure("Consistencia del estimador", constrained_layout=True)
plt.subplot(1,2,1)
plt.title("Sesgo")
plt.plot(N, sesgo, marker='.')
plt.xlabel('número de ventanas K')
plt.ylabel("Sesgo")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(sesgo)-0.01, max(sesgo)+0.01)
plt.grid()

plt.subplot(1,2,2)
plt.title("Varianza ")
plt.plot(N, varianza, marker='.')
plt.xlabel('número de ventanas K')
plt.ylabel("Varianza")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.ylim(min(varianza)-0.01, max(varianza)+0.01)
plt.grid()
plt.tight_layout()


tus_resultados_per = [ 
                   [sesgo0, varianza0], # <-- acá debería haber numeritos :)
                   [sesgo1, varianza1], # <-- acá debería haber numeritos :)
                   [sesgo2, varianza2], # <-- acá debería haber numeritos :)
                   [sesgo3, varianza3], # <-- acá debería haber numeritos :)
                   [sesgo4, varianza4], # <-- acá debería haber numeritos :)
                   [sesgo5, varianza5], # <-- acá debería haber numeritos :)
                   [sesgo6, varianza6], # <-- acá debería haber numeritos :)
                 ]
df = pd.DataFrame(tus_resultados_per, columns=['$s_P$', '$v_P$'],
               index=N)

print(df)
