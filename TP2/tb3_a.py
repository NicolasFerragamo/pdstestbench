#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:01:37 2019

@author: nico
"""
import sys
sys.path.append('/home/nico/Documentos/facultad/6to_nivel/pds/git/pdstestbench')
import os
import matplotlib.pyplot as plt
import numpy as np
import statistics as stats
import scipy.signal as sg
import seaborn as sns
import pandas as pd

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

N  = 1000 # muestras
fs = 2*np.pi # Hz
df = fs / N
a0 = 2 # Volts
p0 = 0 # radianes
f0 = np.pi / 2
Nexp = 500

a = -2 * df
b = 2 * df

#%% generación de frecuencias aleatorias
fa = np.random.uniform(a, b, size = (Nexp)) # genera aleatorios

plt.figure("histograma de frecuencias aleatorias")
plt.hist(fa, bins=20, alpha=1, edgecolor = 'black',  linewidth=1)
plt.ylabel('frequencia')
plt.xlabel('valores')
plt.title('Histograma Uniforme')
plt.savefig("Histograma.png")
plt.show()

#%% generación de señales
f1 = f0 + fa
del fa     

ventanas = [sg.boxcar(N), np.bartlett(N), np.hanning(N), np.blackman(N),  sg.flattop(N)]
V =  len(ventanas)
ventana = ["Rectangular",'Barlett',"Hanning", "Blackman",  "Flattop"]
sesgo = np.zeros((V))
a_est = np.zeros((Nexp, V))
a_mean = np.zeros((V))
varianza = np.zeros((V))
tt = np.linspace(0, (N-1)/fs, N)     


for (ii, this_w) in zip(range(V), ventanas):
     signal = np.vstack(np.transpose([a0 * np.sin(2*np.pi*j*tt) * this_w  for j in f1]))  
    
     mod_signal = np.vstack(np.transpose([np.abs(np.fft.fft(signal[:,ii]))*2/N  for ii in      range(Nexp)]))

     mod_signal = mod_signal[0:int(N/2)]

     a_est[:,ii] = mod_signal[int(N/4)]

     a_mean[ii] = stats.mean(a_est[:,ii])

     sesgo[ii] = a_mean[ii] - a0

     varianza[ii] = stats.variance(a_est[:, ii])
     
     hist, bin_edges = np.histogram(a_est[:,ii], density=True)
     
     # error relativo 
     error_relativo = np.abs((a_est[:,ii] - a0) / a0)
    
     
     #grafico de las ventanas
     fig = plt.figure("Estimar la amplitud con ventana " + ventana[ii], constrained_layout=True)
     gs = fig.add_gridspec(1, 2)

     #gráfico de los errores relativos
     f_ax1 = fig.add_subplot(gs[0, 0])
     f_ax1.set_title("error relativo al estimar la amplitud con la ventana " + ventana[ii] )
     f_ax1.plot(f1, error_relativo, marker='.', linestyle="none")
     f_ax1.set_xlabel('frecuencia normalizada [f/fs]')
     f_ax1.set_ylabel("error de amplitud")
     f_ax1.axhline(0, color="black")
     f_ax1.axvline(0, color="black")
     f_ax1.set_xlim(min(f1), max(f1))
     f_ax1.set_ylim(min(error_relativo)-0.01, max(error_relativo)+0.01)
     f_ax1.grid()

     #grafico del modulo la ventana de python
     f_ax2 = fig.add_subplot(gs[0, 1])
     f_ax2.set_title("Histograma de a_est con ventana " + ventana[ii] )
     f_ax2.hist(a_est[:,ii], bins=20, alpha=1, edgecolor = 'black',  linewidth=1, label=ventana[ii])
     f_ax2.set_xlabel('valores')
     f_ax2.set_ylabel('frecuencia')
     f_ax2.axhline(0, color="black")
     f_ax2.axvline(0, color="black")
     f_ax2.grid()
          
     #conjunto de histogramas para comarar
     plt.figure("Histograma de a_est con ventana "  )
     plt.hist(a_est[:,ii], bins=20, alpha=1, edgecolor = 'black',  linewidth=1, label=ventana[ii])
     plt.legend(loc = 'upper right')
     plt.ylabel('frecuencia')
     plt.xlabel('valores')
     plt.title('histograma de errores al estimar amplitud')
     plt.show()         


tus_resultados = np.vstack(np.transpose([sesgo,varianza]))
df = pd.DataFrame(tus_resultados, columns=['$s_a$', '$v_a$'],
               index=[  
                        'Rectangular',
                        'Bartlett',
                        'Hann',
                        'Blackman',
                        'Flat-top'
                     ])

print(df)