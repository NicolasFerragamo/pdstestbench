#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:38:56 2019

@author: nico
"""

#%% importo los paquetes necesarios
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics as stats

import sys
sys.path.append('/home/nico/Documentos/facultad/6to_nivel/pds/git/pdstestbench')
from pdsmodulos.signals import signals as sg 
from pdsmodulos.signals import FFT

#%% limpio el entorno
os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

#%% Estalesco los datos necesarios
N  = 1000 # muestras
fs = 1000 # Hz
df = fs / N
a0 = 1 # Volts
p0 = 0 # radianes
f0 = df# Hz

#%% Genero las variables necesarias
signal = np.zeros(N)
ruido  = np.zeros(N)

energia_frecuencia4  = 0
energia_frecuencia8  = 0
energia_frecuencia16 = 0

energia_q4  = 0
energia_q8  = 0
energia_q16 = 0

energia_e4  = 0
energia_e8  = 0
energia_e16 = 0


#%% generacion y muestreo de las senoidal

tt, signal = sg.seno(fs, f0, N, a0, p0)
tt, ruido  = sg.noise (fs, f0, N, a0, SNR=10.0) 
SR = signal + ruido

#%% cuntifico las señales
SQ4  = sg.quantizer(SR, 4)
SQ8  = sg.quantizer(SR, 8)
SQ16 = sg.quantizer(SR, 16)

#%% calculo el error de cuantificacion

e4  = SR - SQ4
e8  = SR - SQ8
e16 = SR - SQ16

#%% FFT 

fftSR      = np.fft.fft(SR)
mod_fftSR  = abs(fftSR)

# FFT señales quantizadas
fftSQ4      = np.fft.fft(SQ4)
mod_fftSQ4  = abs(fftSQ4)
fftSQ8      = np.fft.fft(SQ8)
mod_fftSQ8  = abs(fftSQ8 )
fftSQ16     = np.fft.fft(SQ16)
mod_fftSQ16 = abs(fftSQ16)

#FFT señales de error
ffte4      = np.fft.fft(e4)
mod_ffte4  = abs(ffte4)
ffte8      = np.fft.fft(e8)
mod_ffte8  = abs(ffte8)
ffte16     = np.fft.fft(e16)
mod_ffte16 = abs(ffte16)


#%% Cálculo de las energías

# Energías frecuenciales de las señales con ruido

for ii in range(0,N):
     energia_frecuencia4  += mod_fftSR[ii]**2
energia_frecuencia4 /= N**2
     
for ii in range(0,N):
     energia_frecuencia8 += mod_fftSR[ii]**2
energia_frecuencia8 /= N**2
     
for ii in range(0,N):
     energia_frecuencia16 += mod_fftSR[ii]**2
energia_frecuencia16 /= N**2
     

# Energías de la señal cuantizadas
     
for ii in range(0,N):
     energia_q4 += mod_fftSQ4[ii]**2
energia_q4 /= N**2
     
for ii in range(0,N):
     energia_q8 += mod_fftSQ8[ii]**2
energia_q8 /= N**2
     
for ii in range(0,N):
     energia_q16 += mod_fftSQ16[ii]**2
energia_q16 /= N**2
    

# Energías de la señal de error
     
for ii in range(0,N):
     energia_e4 += mod_ffte4[ii]**2
energia_e4 /= N**2
    
for ii in range(0,N):
     energia_e8 += mod_ffte8[ii]**2
energia_e8 /= N**2
     
for ii in range(0,N):
     energia_e16 += mod_ffte16[ii]**2
energia_e16 /= N**2

resultadosSR = [energia_frecuencia4, energia_frecuencia8, energia_frecuencia16]
resutladosSq = [energia_q4, energia_q8, energia_q16]
resultadose  = [energia_e4, energia_e8, energia_e16]


#%% Cálculo del valor medidio de e*
valor_medio_e4 = 0
for ii in range(0,N) : 
     valor_medio_e4 += e4[ii]
valor_medio_e4 /= N

valor_medio_e8 = 0
for ii in range(0,N) : 
     valor_medio_e8 += e8[ii]
valor_medio_e8 /= N

valor_medio_e16 = 0
for ii in range(0,N) : 
     valor_medio_e16 += e16[ii]
valor_medio_e16 /= N

#%% Cálculo del valor RMS
valor_RMS_e4 = 0
for ii in range(0,N) : 
     valor_RMS_e4 += e4[ii]**2
valor_RMS_e4 /= N
valor_RMS_e4 = np.sqrt(valor_RMS_e4)

valor_RMS_e8 = 0
for ii in range(0,N) : 
     valor_RMS_e8 += e8[ii]**2
valor_RMS_e8 /= N
valor_RMS_e8 = np.sqrt(valor_RMS_e8)

valor_RMS_e16 = 0
for ii in range(0,N) : 
     valor_RMS_e16 += e16[ii]**2
valor_RMS_e16 /= N
valor_RMS_e16 = np.sqrt(valor_RMS_e16)


#%% Cálculos de estadísticos

# valor medio
vm_e4  = stats.mean(e4)
vm_e8  = stats.mean(e8)
vm_e16 = stats.mean(e16)

# varianza
var_e4  = stats.variance(e4)
var_e8  = stats.variance(e8)
var_e16 = stats.variance(e16)

# desvio estanda
dstd_e4  = stats.stdev(e4)
dstd_e8  = stats.stdev(e8)
dstd_e16 = stats.stdev(e16)

resultado_vm   = [vm_e4, vm_e8, vm_e16]
resultado_var  = [var_e4, var_e8, var_e16]
resultado_dstd = [dstd_e4, dstd_e8, dstd_e16]


#%% Gŕaficos de las señales en tiempo
# señal senoidal
plt.figure("Gráfico de la señal temporal")
plt.subplot(3,1,1)
plt.plot(tt, signal, 'b', label='f0 = 1Hz ')
plt.xlabel('tiempo [S]')
plt.ylabel('Amplitud [UA]')
plt.grid()
plt.title('Gráfico de la señal temporal sin ruido')
plt.legend(loc = 'upper right')

#Ruido
plt.subplot(3,1,2)
plt.plot(tt, ruido, 'b', label='f0 = 1Hz y SNR=10')
plt.xlabel('tiempo [S]')
plt.ylabel('Amplitud [UA]')
plt.grid()
plt.title('Gráfico el ruido con SNR = 10')
plt.legend(loc = 'upper right')

# senoidal con ruido
plt.subplot(3,1,3)
plt.plot(tt, SR, 'b', label='f0 = 1Hz  y SNR =10')
plt.xlabel('tiempo [S]')
plt.ylabel('Amplitud [UA]')
plt.grid()
plt.title('Gráfico de la señal temporal con ruido con SNR=10')
plt.legend(loc = 'upper right')

plt.tight_layout()


#%% Gráficos de la señal cuantizada

plt.figure("Gráfico de la señal cuantizada")
plt.subplot(3,1,1)
plt.plot(tt, SQ4, 'b', label='SQ4')
plt.xlabel('tiempo [S]')
plt.ylabel('Amplitud [UA]')
plt.grid()
plt.title('Gráfico de la señal cuantizada con SQ4')
plt.legend(loc = 'upper right')

#Ruido
plt.subplot(3,1,2)
plt.plot(tt, SQ8, 'b', label='SQ8')
plt.xlabel('tiempo [S]')
plt.ylabel('Amplitud [UA]')
plt.grid()
plt.title('Gráfico de la señal cuantizada con SQ8')
plt.legend(loc = 'upper right')


# senoidal con ruido
plt.subplot(3,1,3)
plt.plot(tt, SQ16, 'b', label='SQ16')
plt.xlabel('tiempo [S]')
plt.ylabel('Amplitud [UA]')
plt.grid()
plt.title('Gráfico de la señal cuantizada con SQ16')
plt.legend(loc = 'upper right')


plt.tight_layout()


#%% Grafico de los errores

plt.figure("Gráfico de la señal de error")
plt.plot(tt, e4, 'b', label=' SQ = 4')
plt.plot(tt, e8, 'r', label=' SQ = 8')
plt.plot(tt, e16, 'g', label=' SQ = 16')
plt.xlabel('tiempo [S]')
plt.ylabel('Amplitud [UA]')
plt.grid()
plt.title('Gráfico de las señales de error')
plt.legend(loc = 'upper right')

#%% Gráficos de los histogramas de los errores

#%% Ploteo de los errores

plt.figure("Histograma de errores de cuantificación ")
plt.hist(e4, bins=50, alpha=1, edgecolor = 'black',  linewidth=1, label="e4")
plt.hist(e8, bins=50, alpha=1, edgecolor = 'black',  linewidth=1, label="e8")
plt.hist(e16, bins=50, alpha=1, edgecolor = 'black',  linewidth=1, label="416")
plt.legend(loc = 'upper right')
plt.ylabel('frecuencia')
plt.xlabel('valores')
plt.title('histograma de errores de cuantificación')
plt.show()


#%% tabla de panda
tus_resultados = [ ['$\sum_{f=0}^{f_S/2} \lvert S_R(f) \rvert ^2$', '$\sum_{f=0}^{f_S/2} \lvert S_Q(f) \rvert ^2$', '$\sum_{f=0}^{f_S/2} \lvert e(f) \rvert ^2$' ], 
                   ['',                                             '',                                             ''                              ], 
                   [resultadosSR[0], resutladosSq[0], resultadose[0]], # <-- completar acá
                   [resultadosSR[1], resutladosSq[1], resultadose[1]], # <-- completar acá
                   [resultadosSR[2], resutladosSq[2], resultadose[2]], # <-- completar acá
                 ]

df = pd.DataFrame(tus_resultados, columns=['Energía total', 'Energía total Q', 'Energía total $e$'],
               index=['$f_0$ \ expr. matemática', 
                      '', 
                      '4 bits', 
                      '8 bits', 
                      '16 bits'
                      ])

print(df)