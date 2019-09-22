#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:56:13 2019

@author: nico
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import cmath 

#%% Lipieza de gráficos

#os.system ("clear") # limpia la terminal de python
#plt.close("all")    #cierra todos los graficos 



#%%  TRABAJAMOS CON LA FFT

#  Esta funcion recive los la fft de la señal y se encarga de plotearla controlando varios parámetros
#  y1l: etiqueta vertical del módulo
#  y2l: etiqueta vertical de la fase
#  p1t: título del modulo
#  p2t: título de la fase
#  tp: título de la figura
#  loc1: localización de las etiquetas en el módulo
#  loc2: localización de las etiquetas en la fase
#  c: color del gráfico por defecto es rojo  ['r', 'b', 'g', 'c', 'm', 'y', 'k']
#  l: nombre de la etiqueta
#  db: grafica el modulo en dB si esta en 'ON' o en veces si esta en 'off' por defecto esta actiado
#  tipo:  determina si quiero usar plot o stem por defecto esta activado plot
#  m:  marker por defecto esta '-'
#  ls: Linestyle  por defecto = 'None' (es la interpolacion)
#  col_ax: activa o desactiva el color de los ejes por defecto esta encendido

def plotFFT (fftsignal, fs, N, y1l='Amplitud Normlizada [db] ', y2l='Fase [rad] ', p1t=' ',
             p2t=' ', tp="FFT de la señal", loc1='upper right', loc2='upper right', c=0,
             l=' ', db='ON', tipo='plot', m='.',ls='None', col_ax = 'on') :
    
    mod_signal, fase_signal = Mod_and_Angle_signal (fftsignal, db)
    N =len(fftsignal)
    df= fs / N
    col= ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    
    
    
#%% Ploteo de la FFT
    plt.figure(tp)
    plt.subplot(2,1,1)
    freq = np.linspace(0, (N-1)*df, N) / fs
    if tipo == 'stem':
        plt.stem(freq[0:int(N/2)], mod_signal[0:int(N/2)], col[c], label='modulo '+ l,
                  marker=m, linestyle=ls)
    else:
         plt.plot(freq[0:int(N/2)], mod_signal[0:int(N/2)], col[c], label='modulo '+ l,
                   marker=m, linestyle=ls)
    plt.xlabel('frecuecnia normalizada f/fs [Hz]')
    plt.ylabel(y1l)
    if col_ax == 'ON'or col_ax == 'on' :
         plt.axhline(0, color="black")
         plt.axvline(0, color="black")
    #plt.xlim((0.2,0.3))
    plt.grid()
    plt.title('Modulo de la señal '+p1t)
    plt.legend(loc = loc1)


    plt.subplot(2,1,2)
    if tipo == 'stem':
        plt.stem(freq[0:int(N/2)], fase_signal[0:int(N/2)], col[c], label='fase '+ l, 
                   marker=m, linestyle=ls)
    else:
        plt.plot(freq[0:int(N/2)], fase_signal[0:int(N/2)], col[c], label='fase '+ l, 
                    marker=m, linestyle=ls)
    plt.xlabel('frecuecnia normalizada f/fs [Hz]')
    plt.ylabel(y2l)
    if col_ax == 'ON'or col_ax == 'on' :
         plt.axhline(0, color="black")
         plt.axvline(0, color="black")
    plt.grid()
    plt.title('fase de la señal '+p2t)
    plt.legend(loc = loc2)
    plt.tight_layout() #para ajustar el tamaño de lo contrario se puperpinan los titulos
    plt.show()
    
    return 0



#%% my DFT
# Esta función realiza la DFT de una señal de tamaño N
# solo necesita como parámetro la señal a transformar
    

def myDFT (signal) :
    
    N =len(signal)
    Signal = np.empty(N)
    Signal[:N-1] = np.nan
    W = [ ]
    W = np.zeros((N,N),dtype=complex)  # tengo que limpiar la memoria, el vector
    for k in range (0, N-1):
        for n in range (0, N-1):
            W[k][n] = cmath.exp(-1j * 2 * np.pi * k * n/N)  #  calcula los twiddles factors
    Signal = np.dot(W,  signal)  #  Realiza la multiplicación punto a punto de la señal 
    return Signal


#%% convierto la señal en dB y la normalizo
#  Separa la señal transformada en módulo y fase
#  Por defecto retorna el módulo en dB y normalizado, de no quererlo en db utilizar db='off'    
    

def Mod_and_Angle_signal (fftsignal, db='ON') :
    
    N =len(fftsignal)
    mod_signal = np.abs(fftsignal) *2 / N
    fase_signal = np.angle(fftsignal)
    if db == 'ON'or db == 'on' :
        mod_signal = 20 *np.log10(mod_signal)
        
    return mod_signal, fase_signal