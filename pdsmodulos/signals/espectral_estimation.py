#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:11:01 2019

@author: nico
"""
from scipy.fftpack import fft
import numpy as np
"""
brief: Esta funcio realiza el periodograma de una señal 

argumentos

signal: es la señal de entrada a la cual se le va a realizar el periodogram tiene 
que ser un vector columna, de lo contrario aclarar en ax=1 para vector fila

n1: es la psosision inicial
n2: es la posision final
Sper: retorna el periodrograma de salida
"""

def periodogram(signal, n1=0, n2=0, ax=0):
     
    if n1 == 0 and n2 == 0 : # por defecto usa la selal completa
        n1=0;
        n2=len(signal)
    return  1/(n2 - n1) *(np.abs(fft(signal[n1:n2], axis=ax)))**2


"""
brief: Esta funcio realiza el periodograma modificado de una señal 

argumentos

signal: es la señal de entrada a la cual se le va a realizar el periodogram tiene 
que ser un vector columna, de lo contrario aclarar en ax=1 para vector fila
vent: ventana a utilizar
n1: es la psosision inicial
n2: es la posision final
ax: es el eje de la fft si es 0 hace vector columna, si es uno vector fila
Sper: retorna el periodrograma de salida
"""

def mperiodogram(signal, vent='Barlet', n1=0, n2=0, ax=0):

     
     
    if n1 == 0 and n2 == 0 : # por defecto usa la selal completa
        n1=0;
        n2=len(signal)
    return  1/(n2 - n1) *(np.abs(fft(signal[n1:n2], axis=ax)))**2