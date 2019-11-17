#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:11:01 2019

@author: nico
"""
from scipy.fftpack import fft
from scipy.signal import correlate
import scipy.signal as sg
import numpy as np
import sys

"""
brief: Esta funcio realiza el periodograma de una señal 

argumentos

signal: es la señal de entrada a la cual se le va a realizar el periodogram tiene 
que ser un vector columna, de lo contrario aclarar en ax=1 para vector fila

n1: es la psosision inicial
n2: es la posision final
Sper: retorna el periodrograma de salida
"""

def periodogram(signal, n1=0, n2=0, exp=0, ax=0):
     
    if n1 == 0 and n2 == 0 : # por defecto usa la selal completa
        n1=0;
        n2=len(signal)
    N = n2 - n1    
    if exp > 1 :
         return 1/N *(np.abs(fft(signal[n1:n2,:], axis=ax)))**2
    else :
         return  1/N *(np.abs(fft(signal[n1:n2], axis=ax)))**2


"""
brief: Esta funcio realiza el periodograma modificado de una señal 

argumentos

signal: es la señal de entrada a la cual se le va a realizar el periodogram tiene 
que ser un vector columna, de lo contrario aclarar en ax=1 para vector fila
vent: ventana a utilizar
n1: es la psosision inicial
n2: es la posision final
ax: es el eje de la fft si es 0 hace vector columna, si es uno vector fila
Sper: retorna el periodrograma modificado de salida
win:  ventana a utililizar [Bartlet, Hanning, Hamming, Blackman, Flattop] También se le puede enviar una ventana
exp: número dvente experimentos para trabajar de forma matricial
"""

def mperiodogram(signal, win='Bartlett', n1=0, n2=0, exp=0, ax=0):
   
    if n1 == 0 and n2 == 0 : # por defecto usa la selal completa
        n1=0;
        n2=len(signal)
    N = n2 - n1
    
    if win == "Bartlett":
         w = np.bartlett(N)
    elif win == "Hanning" :
         w = np.hanning(N)
    elif win == "Hamming":   
         w = np.hamming(N)
    elif win == "Blackman":   
         w = np.blackman(N)  
    elif win == "Flattop" : 
         w = sg.flattop(N)    
    else :
         w = sg.boxcar(N) 
         
    aux = signal[n1:n2] * w 
    U = sum(np.abs(w)**2)/ N
    return periodogram(aux, exp, ax=ax) / U

            
"""
brief: Esta funcio realiza el metodo de Barlett de una señal 

argumentos

signal: es la señal de entrada a la cual se le va a realizar el periodogram tiene 
que ser un vector columna, de lo contrario aclarar en ax=1 para vector fila
k: numero de ventanas
ax: es el eje de la fft si es 0 hace vector columna, si es uno vector fila
Sper: retorna el barlett de salida
vent:  ventana a utililizar [Barlet, Hanning, Hamming, Blackman, Flattop] También se le puede enviar una ventana
exp: número dvente experimentos para trabajar de forma matricial
"""

def barlett(signal, K, ax=0):
      
     N = len(signal)
     L = int(np.floor(N / K)) # calculo la longitud de la ventana
     n1 = 0
     Px = 0
     for i in range(K) : # calcula un periodograma por cada sector
          Px = Px + periodogram(signal, n1=n1, n2=n1+L, exp=0, ax=ax)
          # voy sumando los K Px y los promedio
          n1 = n1 + L
     return Px / K

"""
brief: Esta funcio realiza el metodo de Welch de una señal 

argumentos

signal: es la señal de entrada a la cual se le va a realizar el periodogram tiene 
que ser un vector columna, de lo contrario aclarar en ax=1 para vector fila
k: numero de ventanas
ax: es el eje de la fft si es 0 hace vector columna, si es uno vector fila
Sper: retorna el welch de salida
win:  ventana a utililizar [Barlet, Hanning, Hamming, Blackman, Flattop] por defecto ventana rectangular
exp: número dvente experimentos para trabajar de forma matricial
L longitud de la ventena
over:  overlap
N=K*L

"""

def welch(signal, L, over=0.5, win="Bartlett", ax=0):
     
     if over >=1 or over < 0 :
          sys.stderr.write("overlap is invalid \n  default over=0.5\m")
          over = 0.5
     L = int(L)     
     D  = int((1-over)*L)  # desplazamiento
     K  = int(np.floor((len(signal)-L)/ D))+1 # calculo del número de ventanas
     n1 = 0
     Px = 0
     for i in range(K) : # calcula un periodograma por cada sector
          Px = Px + mperiodogram(signal, win=win, n1=n1, n2=n1+L, exp=0, ax=ax)
          # voy sumando los K Px y los promedio
          n1 = n1 + D
     return Px / K



"""
brief: Esta funcio realiza el metodo de blackman-tukey de una señal 

argumentos

signal: es la señal de entrada a la cual se le va a realizar el periodogram tiene 
que ser un vector columna, de lo contrario aclarar en ax=1 para vector fila

ax: es el eje de la fft si es 0 hace vector columna, si es uno vector fila
Sper: retorna el blackman-tukey de salida
win:  ventana a utililizar [Barlet, Hanning, Hamming, Blackman, Flattop] por defecto ventana rectangular


"""

def blakmanTukey(signal, M=0, win="Bartlett", n1=0, n2=0, ax=0):
     
          
     if n1 == 0 and n2 == 0 : # por defecto usa la selal completa
        n1=0;
        n2=len(signal)
        
     N = n2 - n1
     if M == 0 :
          M = int(N/5)
          
     M = 2*M-1
     if M > N:
          raise ValueError('Window cannot be longer than data')
          
     if win == "Bartlett":
         w = np.bartlett(M)
     elif win == "Hanning" :
         w = np.hanning(M)
     elif win == "Hamming":   
          w = np.hamming(M)
     elif win == "Blackman":   
         w = np.blackman(M)  
     elif win == "Flattop" : 
         w = sg.flattop(M)    
     else :
         w = sg.boxcar(M) 
         
     r, lags = acorrBiased(signal)    
     r = r[np.logical_and(lags >= 0, lags < M)]
     rw = r * w
     Px = 2 * fft(rw).real - rw[0];
     
     return Px   
    
    
def acorrBiased(y):
  """Obtain the biased autocorrelation and its lags
  """
  r = correlate(y, y) / len(y)
  l = np.arange(-(len(y)-1), len(y))
  return r,l
  