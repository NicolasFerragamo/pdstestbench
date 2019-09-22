#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:32:09 2019

@author: nico
"""


import numpy as np
import scipy.signal as scsg

#%% senoidal

"""
    brief:  Generador de señales senoidal, con argumentos
    
    fs:     frecuencia de muestreo de la señal [Hz]
    N:      cantidad de muestras de la señal a generar
    f0:     frecuencia de la senoidal [Hz]
    a0:     amplitud pico de la señal [V]
    p0:     fase de la señal sinusoidal [rad]
    
    como resultado la señal devuelve:
    
    signal: senoidal evaluada en cada instante 
    tt:     base de tiempo de la señal
    """    

def seno (fs=1000, f0=1, N=1000, a0=2, p0=0) :
    w = 2 * np.pi * f0
    tt = np.linspace(0, (N-1)/fs, N)  
    signal = a0  * np.sin(w * tt + p0)
    return tt, signal


#%% square
    
"""
    brief:  Generador de señales cuadrada, con argumentos
    
    fs:     frecuencia de muestreo de la señal [Hz]
    N:      cantidad de muestras de la señal a generar
    f0:     frecuencia de la senoidal [Hz]
    a0:     amplitud pico de la señal [V]
    p0:     fase de la señal sinusoidal [rad]
    dt:     ciclo de trabajo de la señal [0-1]
    como resultado la señal devuelve:
    
    signal: senoidal evaluada en cada instante 
    tt:     base de tiempo de la señal
    """    
    
def square (fs=1000, f0=1, N=1000, a0=2, p0=0, duty=50) :
    w = 2 * np.pi * f0
    duty = duty / 100 
    tt = np.linspace(0, (N-1)/fs, N)  
    signal = a0  * scsg.square(w * tt + p0, duty )
    return tt, signal

#%% sawtooth
    
"""
    brief:  Generador de señales diente de sierra, con argumentos
    
    fs:     frecuencia de muestreo de la señal [Hz]
    N:      cantidad de muestras de la señal a generarsignal = []
signal = np.zeros((N,3))
    f0:     frecuencia de la senoidal [Hz]
    a0:     amplitud pico de la señal [V]
    p0:     fase de la señal sinusoidal [rad]
    width:  ancho de la señal [0-1]
    como resultado la señal devuelve:
    
    signal: senoidal evaluada en cada instante 
    tt:     base de tiempo de la señal
    """    
    
def sawtooth (fs=1000, f0=1, N=1000, a0=2, p0=0, width=50) :
    w = 2 * np.pi * f0
    width = width / 100 
    tt = np.linspace(0, (N-1)/fs, N)  
    signal = a0  * scsg.sawtooth(w * tt + p0, width )
    return tt, signal


#%% noise
    
"""
    brief:  Generador de señales senoidal, con argumentos
    
    fs:     frecuencia de muestreo de la señal [Hz]
    N:      cantidad de muestras de la señal a generar
    f0:     frecuencia de la senoidal [Hz]
    a0:     amplitud pico de la señal [V]
    SNR:    relación señal ruido en veces
    varianza: varianza de la señal de ruido (opcional)
    
    como resultado la señal devuelve:
    
    signal: señal de ruido con distribución normal
    tt:     base de tiempo de la señal
    """    
def noise (fs=1000, f0=1, N=1000, a0=1, SNR=1000.0, varianza = 'None') :
    
    pot_senoidal = (a0**2) / 2 
    if varianza != 'None' :
         ruido = varianza * np.random.normal(0, 1, N)
    else :
         varianza_ruido = pot_senoidal/pow(10, SNR/10)
         ruido = varianza_ruido * np.random.normal(0, 1, N)
    tt = np.linspace(0, (N-1)/fs, N)
    return tt, ruido


#%% quantizer
    
    """
    brief:  Cuantificador
    
    signal:  señal a cuantificar
    n:       # de bits de cuantificacion
    
    como resultado la señal devuelve:
    
    qsignal: señal cuantificada
    """    
def quantizer (signal, n) :
     
     qsignal=np.round(signal*(2**n)/2) / ((2**n)/2)
     return qsignal


"""
    brief:  Cuantificador con limitador unitario
    
    signal:  señal a cuantificar
    n:       # de bits de cuantificacion
    
    como resultado la señal devuelve:
    
    qsignal: señal cuantificada
"""    
# esta función solo cuantifica hasta 1 si la señal se pasa la aplana en 1, tienen un limitador

def uniform_midtread_quantizer(signal, n):
     
    Q = (max(signal) - min(signal)) / 2**n
    # limiter
    signal = np.copy(signal)
    idx = np.where(np.abs(signal) >= 1)
    signal[idx] = np.sign(signal[idx])
    # linear uniform quantization
    qsignal= Q * np.floor(signal/Q + 1/2)

    return qsignal
