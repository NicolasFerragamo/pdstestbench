#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:32:09 2019

@author: nico
"""
import numpy as np




def seno (fs, f0, N, a0=1, p0=0) :
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
    
    w  = 2 * np.pi * f0
    tt = np.arange(0, (N - 1)/fs, 1 / fs )
    signal = a0  * np.sin(w * tt + p0)
    return (tt, signal)

