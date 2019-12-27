#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:20:47 2019

@author: nico
"""
import os
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('/home/nico/Documentos/facultad/6to_nivel/pds/git/pdstestbench')
from pdsmodulos.signals import signals as sg 
from pdsmodulos.signals import FFT

os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

N  = 1000 # muestras
fs = 1000 # Hz
df = fs / N
a0 = 1 # Volts
p0 = 0 # radianes
f0 = 100 # Hz
w  = 2 * np.pi * f0
duty = 50
width = 100

lsignal = ['seno', 'cuadrada', 'Dientes de cierra']

signal = []
signal = np.zeros((N,3),dtype='complex')
fftsignal= []
fftsignal= np.zeros((N,3),dtype='complex')
fftsignal1= []
fftsignal1= np.zeros((N,3),dtype='complex')

tt,signal[:,0] = sg.seno (fs, f0, N, a0, p0)
tt,signal[:,1] = sg.square (fs, f0, N, a0, p0, duty) 
tt,signal[:,2] = sg.sawtooth (fs, f0, N, a0, p0, width) 

ii = 0
while ii < 3 : 
    fftsignal[:,ii] = np.fft.fft(signal[:,ii])
    FFT.plotFFT(fftsignal[:,ii],fs,N, tp= 'FFT', c=ii, db='off', l=lsignal[ii])
    fftsignal1[:,ii] = FFT.myDFT(signal[:,ii])
    FFT.plotFFT(fftsignal1[:,ii],fs,N, tp= 'DFT', c=ii, db='off', l=lsignal[ii])
    ii += 1