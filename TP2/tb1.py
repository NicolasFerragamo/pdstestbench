#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 19:15:01 2019

@author: nico
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg
from pdsmodulos.signals import FFT
from pdsmodulos.signals import windows as wds

#import seaborn as sns

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

N  = 1000 # muestras
fs = 1000 # Hz
df = fs / N
a0 = 2 # Volts
p0 = 0 # radianes
f0 = fs / 4

M = 9*N

ventanas = [np.bartlett(N), wds.triang(N), np.hanning(N), wds.hann(N),    np.blackman(N), wds.blackamanHarris(N), sg.flattop(N), wds.flattop(N), sg.boxcar(N), sg.boxcar(N)]
V =  len(ventanas)
ventana = ["Bartell", "my_Triang", "Hanning", "my_Hanning", "Blackman", "my_Blackman", "Flattop", "my_Flattop", "Rectangular", "my_Rectangular"]

aux_padding = np.zeros(M, float)

# creo las ventanas y obtengo el móduo de su dft
signal = np.vstack(np.transpose([this_w  for this_w in ventanas])) 
mod_signal = np.vstack(np.transpose([np.abs(np.fft.fft(signal[:,ii]))*2/N  for ii in      range(V)]))

# creo las ventanas con zero padding y obtengo el móduo de su dft
signal_padding = np.vstack(np.transpose([np.concatenate((this_w,aux_padding), axis=0)  for this_w in ventanas])) 
mod_signal_padding = np.vstack(np.transpose([np.abs(np.fft.fft(signal_padding[:,ii]))*2/(N + M)  for ii in      range(V)]))

# la paso en dB
mod_signal = 20 *np.log10(mod_signal/mod_signal[0])
mod_signal_padding = 20 *np.log10(mod_signal_padding/mod_signal_padding[0])


#Genero los ejes de tiempo y frecuencia
tt = np.linspace(0, (N-1)/fs, N)
freq = np.linspace(0, (N-1)*df, N) / fs
freq2 = np.linspace(0, (N + M-1)*df, N + M) / (10*fs)

for ii in (0,2,4,6,8):

#grafico de las ventanas
     fig = plt.figure("ventana " + ventana[ii], constrained_layout=True)
     gs = fig.add_gridspec(2, 3)

#grafico de la ventana de python
     f_ax1 = fig.add_subplot(gs[0, 0])
     f_ax1.set_title(ventana[ii])
     f_ax1.plot(tt,ventanas[ii])
     f_ax1.set_xlabel("tiempo [S]")
     f_ax1.set_ylabel("Amplitud")
     f_ax1.axhline(0, color="black")
     f_ax1.axvline(0, color="black")
     f_ax1.grid()

#grafico del modulo la ventana de python
     f_ax2 = fig.add_subplot(gs[0, 1])
     f_ax2.set_title("FFT " + ventana[ii])
     f_ax2.plot(freq[0:int(N/2)], mod_signal[0:int(N/2),ii], marker='.', linestyle='None')
     f_ax2.set_xlabel('frecuecnia normalizada f/fs [Hz]')
     f_ax2.set_ylabel("Magnitud [dB]")
     f_ax2.axhline(0, color="black")
     f_ax2.axvline(0, color="black")
     f_ax2.grid()

#grafico del modulo la ventana con zero padding de python
     f_ax3 = fig.add_subplot(gs[0, 2])
     f_ax3.set_title("Zero padding " + ventana[ii])
     f_ax3.plot(freq2[0:int((N+M)/2)], mod_signal_padding[0:int((N + M)/2),ii], marker='.', linestyle='None')
     f_ax3.set_xlabel('frecuecnia normalizada f/fs [Hz]')
     f_ax3.set_ylabel("Magnitud [dB]")
     f_ax3.axhline(0, color="black")
     f_ax3.axvline(0, color="black")
     f_ax3.set_xlim(0, 0.015)
     f_ax3.set_ylim(-150, 0)
     f_ax3.grid()

#grafico de la ventana propia
     f_ax4 = fig.add_subplot(gs[1, 0])
     f_ax4.set_title(ventana[ii + 1])
     f_ax4.plot(tt,ventanas[ii + 1])
     f_ax4.set_xlabel("tiempo [S]")
     f_ax4.set_ylabel("Amplitud")
     f_ax4.axhline(0, color="black")
     f_ax4.axvline(0, color="black")
     f_ax4.grid()

#grafico del modulo la ventana propia
     f_ax5 = fig.add_subplot(gs[1, 1])
     f_ax5.set_title("FFT " + ventana[ii + 1])
     f_ax5.plot(freq[0:int(N/2)], mod_signal[0:int(N/2),ii + 1], marker='.', linestyle='None')
     f_ax5.set_xlabel('frecuecnia normalizada f/fs [Hz]')
     f_ax5.set_ylabel("Magnitud [dB]")
     f_ax5.axhline(0, color="black")
     f_ax5.axvline(0, color="black")
     f_ax5.grid()

#grafico del modulo la ventana con zero padding propia
     f_ax6 = fig.add_subplot(gs[1, 2])
     f_ax6.set_title("Zero padding " + ventana[ii + 1])
     f_ax6.plot(freq2[0:int((N+M)/2)], mod_signal_padding[0:int((N + M)/2),ii + 1], marker='.', linestyle='None')
     f_ax6.set_xlabel('frecuecnia normalizada f/fs [Hz]')
     f_ax6.set_ylabel("Magnitud [dB]")
     f_ax6.axhline(0, color="black")
     f_ax6.axvline(0, color="black")
     f_ax6.set_xlim(0, 0.015)
     f_ax6.set_ylim(-150, 0)
     f_ax6.grid()


fig2 = plt.figure("Comparación de los módulos de las ventanas")
plt.plot(freq2[0:int((N + M)/2)], mod_signal_padding[0:int((N + M)/2),0], marker='.', linestyle='None', label=ventana[0])
plt.plot(freq2[0:int((N + M)/2)], mod_signal_padding[0:int((N + M)/2),2], marker='.', linestyle='None', label=ventana[2])
plt.plot(freq2[0:int((N + M)/2)], mod_signal_padding[0:int((N + M)/2),4], marker='.', linestyle='None', label=ventana[4])
plt.plot(freq2[0:int((N + M)/2)], mod_signal_padding[0:int((N + M)/2),6], marker='.', linestyle='None', label=ventana[6])
plt.plot(freq2[0:int((N + M)/2)], mod_signal_padding[0:int((N + M)/2),8], marker='.', linestyle='None', label=ventana[8])
plt.xlabel('frecuecnia normalizada f/fs [Hz]')
plt.ylabel("Magnitud [dB]")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.xlim(0, 0.01)
plt.ylim(-150, 0)
plt.grid()
plt.legend(loc ='upper right')