#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:41:32 2019

@author: nico
"""

import os
import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.interpolate import CubicSpline
import scipy.io as sio
from time import time

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

fig_sz_x = 14
fig_sz_y = 13
fig_dpi = 80 # dpi

fig_font_family = 'Ubuntu'
fig_font_size = 16


#Diseño de los filtros digitales

# para listar las variables que hay en el archivo
#sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
ecg_one_lead = ecg_one_lead.flatten(1)
cant_muestras = len(ecg_one_lead)
fs = 1000
tt = np.linspace(0, cant_muestras, cant_muestras)

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

# calculo la frecuencia de corte con el 90% de la enrgia
energia=np.zeros((int(L/2)+1))
np.cumsum(Swelch, out=energia)
limfreq = energia < 0.95*energia[-1]
for ii in range(len(limfreq)) :
     if limfreq[ii] == False:
          freq = ii
          break
     
cant_pasadas  = fs/freq
cant_pasadas = np.log2(cant_pasadas)  #porque cada pasada divide a la mitad
cant_pasadas = np.round(cant_pasadas)



nyq_frec = fs / 2

#usar cumsum para estimar las frecuencias
# filter design
ripple = 0.1 # dB
atenuacion = 40. # dB

wp = 40  #Hz podria ser 0.05 pero uso la media geométrica para que sea simétrico
ws = 70 #Hz
gains =np.array([-atenuacion, -ripple])
gains = 10**(gains/20)
frecs = np.array([0.0, ws,  wp, nyq_frec]) / nyq_frec

L = fs/wp
lp_sos_cauer = sig.iirdesign(wp=0.5, ws=0.55, gpass=ripple, gstop=atenuacion, analog=False, ftype= 'ellip', output='sos')
w, h_cauer = sig.sosfreqz(lp_sos_cauer)  # genera la respuesta en frecuencia del filtro sos

w = w / np.pi   # devuelven w de 0 a pi entonces desnormaliza

eps = np.finfo(float).eps
plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi= fig_dpi, facecolor='w', edgecolor='k')
plt.plot(w, 20*np.log10(np.abs(h_cauer + eps)))
#plt.plot(frecs * nyq_frec, 20*np.log10(gains + eps), 'rx', label='plantilla' )
plt.title('FIR diseñado por métodos directos')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Modulo [dB]')
#plt.xscale('log') 
#plt.axis([0, nyq_frec, -60, 5 ]);
plt.grid()
axes_hdl = plt.gca()
plt.show()

decimation = ecg_one_lead

for jj in range(int(cant_pasadas)):
     
     decimation = sig.sosfiltfilt(lp_sos_cauer, decimation)
     aux = np.zeros((int(len(decimation)/2)))
     for ii in range(int(len(decimation)/2)):
     
          aux[ii] = decimation[2*ii]
     decimation = aux


interpolation = decimation
xx = np.linspace(0,len(decimation)-1,len(decimation))
cs = CubicSpline(xx, decimation)


plt.figure(figsize=(6.5, 4))
plt.plot(xx, cs(decimation), label='true')
plt.show()


#np.array([5, 5.2]) *60*fs
#ff = np.linespace
#plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi= fig_dpi, facecolor='w', edgecolor='k')
#plt.plot(tt[0:int(len(tt)/2)], ECG_f_cauer[0:int(len(tt)/2)], label='Cauer') 
#plt.title('ECG filtering ')
#plt.ylabel('Adimensional')
#plt.xlabel('Muestras (#)')
#axes_hdl = plt.gca()
#axes_hdl.legend()
#axes_hdl.set_yticks(())
#plt.show()




#hay una función para decimar sig.decimate

decimation = ecg_one_lead
for jj in range(int(cant_pasadas)):
     decimation = sig.decimate(decimation, 2)
 
     
the_start = time()
median1_dec = sig.medfilt(decimation, 3) #200 ms
median2_dec = sig.medfilt(median1_dec, 7) #600 ms
the_end = time()
tiempodft_dec = the_end - the_start
del the_start, the_end     

interpolation = median2_dec
for jj in range(int(cant_pasadas)):
     interpolation = sig.resample(interpolation,2*len(interpolation))
signal_int = ecg_one_lead - interpolation[0:len(ecg_one_lead)]     

plt.figure("ECG", constrained_layout=True)
plt.title("ECG")
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

