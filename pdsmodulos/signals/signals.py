#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:32:09 2019

@author: nico
"""


import numpy as np
import scipy.signal as sg

#%% senoidal

def seno (fs=1000, f0=1, N=1000, a0=2, p0=0) :
    w = 2 * np.pi * f0
    tt = np.linspace(0, (N-1)/fs, N)  
    signal = a0  * np.sin(w * tt + p0)
    return tt, signal


#%% square
    
def square (fs=1000, f0=1, N=1000, a0=2, p0=0, duty=50) :
    w = 2 * np.pi * f0
    duty = duty / 100 
    tt = np.linspace(0, (N-1)/fs, N)  
    signal = a0  * sg.square(w * tt + p0, duty )
    return tt, signal

#%% sawtooth
    
def sawtooth (fs=1000, f0=1, N=1000, a0=2, p0=0, width=50) :
    w = 2 * np.pi * f0
    width = width / 100 
    tt = np.linspace(0, (N-1)/fs, N)  
    signal = a0  * sg.sawtooth(w * tt + p0, width )
    return tt, signal


#%% noise
    
def noise (fs=1000, f0=1, N=1000, a0=1, SNR=1000.0, varianza = np.nan) :
    
    pot_senoidal = a0 / 2 
    varianza_ruido = pot_senoidal/pow(10, SNR/10)
    ruido = varianza_ruido * np.random.normal(0, 1, N)
    tt = np.linspace(0, (N-1)/fs, N)
    return tt, ruido
