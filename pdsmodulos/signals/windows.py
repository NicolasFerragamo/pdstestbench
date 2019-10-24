#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:59:11 2019

@author: nico
"""
import numpy as np


#%% Ventana trinagunlar
# N tamaño de la ventana
# nventanas numero de ventanas
# esta funcion genera una matriz de ventanas triangulares

def triang (N, nventanas=1):
     triangular = np.zeros((N, nventanas))
     nn = np.linspace(0, N-1, N)
     triang =  ((2/(N - 1)) * ((N + 1)/2 - np.abs(nn - ((N - 1)/2))))
     if nventanas > 1 :
          for ii in range(nventanas):
               triangular[:, ii] = triang
     else : 
          return triang
     return triangular     


#%% Ventana Hann
# N tamaño de la ventana
# nventanas numero de ventanas
# esta funcion genera una matriz de ventanas triangulares

def hann (N, nventanas=1):
     hann = np.zeros((N, nventanas))
     nn = np.linspace(0, N-1, N)
     hanning = 0.5 * (1 - np.cos((2 * np.pi * nn)/(N-1)))
     if nventanas > 1 :
          for ii in range(nventanas):
               hann[:, ii] = hanning
     else:
          return hanning          
     return hann    


#%% Ventana Hamming
# N tamaño de la ventana
# nventanas numero de ventanas
# esta funcion genera una matriz de ventanas triangulares
def hamming (N, nventanas=1):
     hamm = np.zeros((N, nventanas))
     nn = np.linspace(0, N-1, N)
     hamming = 0.54 -  0.46 * (np.cos(2*np.pi* nn/N))
     if nventanas > 1 :
          for ii in range(nventanas):
               hamm[:, ii] = hamming
     else:
          return hamming
     return hamm 


#%% Ventana Flat-Top
# N tamaño de la ventana
# nventanas numero de ventanas
# esta funcion genera una matriz de ventanas triangulares
def flattop (N, nventanas=1):
     
     a0 = 0.21557895
     a1 = 0.41663158
     a2 = 0.277263158
     a3 = 0.083578947
     a4 = 0.006947368
     
     flat = np.zeros((N, nventanas))
     nn = np.linspace(0, N-1, N)
     flattop = a0 - a1 * np.cos((2 * np.pi * nn)/(N - 1)) + a2 * np.cos((4 * np.pi * nn)/(N - 1)) - a3 * np.cos((6 * np.pi * nn)/(N - 1)) + a4 * np.cos((8 * np.pi * nn)/(N - 1))
     if nventanas > 1 :
          for ii in range(nventanas):
               flat[:, ii] = flattop
     else:
          return flattop
     return flat     

#%% Ventana barlett
# N tamaño de la ventana
# nventanas numero de ventanas
# esta funcion genera una matriz de ventanas triangulares

def barlett (N, nventanas=1):
     barlett = np.zeros((N, nventanas))
     nn = np.linspace(0, N-1, N)
     bar =  ((N - 1)/2)  - np.abs(nn - ((N - 1)/2))
     if nventanas > 1 :
          for ii in range(nventanas):
               barlett[:, ii] = bar *2/N
     else:
          return bar
     return barlett 


#%% Ventana blackamanHarris
# N tamaño de la ventana
# nventanas numero de ventanas
# esta funcion genera una matriz de ventanas triangulares
def blackamanHarris (N, nventanas=1):
     
     a0 = 0.35875
     a1 = 0.48829
     a2 = 0.14128
     a3 = 0.01168
     
     blackman = np.zeros((N, nventanas))
     nn = np.linspace(0, N-1, N)
     blackamanHarris = a0 - a1 * np.cos((2 * np.pi * nn)/(N - 1)) + a2 * np.cos((4 * np.pi * nn)/(N - 1)) - a3 * np.cos((6 * np.pi * nn)/(N - 1)) 
     if nventanas > 1 :
          for ii in range(nventanas):
               blackman[:, ii] = blackamanHarris
     else:
          return blackamanHarris
     return blackman     


#%% Ventana blackamanHarris
# N tamaño de la ventana
# nventanas numero de ventanas
# esta funcion genera una matriz de ventanas triangulares
def blackaman(N, nventanas=1):
     
     a0 = 0.42
     a1 = 0.5
     a2 = 0.08
     
     blackman = np.zeros((N, nventanas))
     nn = np.linspace(0, N-1, N)
     black = a0 - a1 * np.cos((2 * np.pi * nn)/(N - 1)) + a2 * np.cos((4 * np.pi * nn)/(N - 1)) 
     if nventanas > 1 :
          for ii in range(nventanas):
               blackman[:, ii] = black
     else : 
          return black
     return blackman     


#%% Ventana blackamanHarris
# N tamaño de la ventana
# nventanas numero de ventanas
# esta funcion genera una matriz de ventanas triangulares
def blackamanNuttall (N, nventanas=1):
     
     a0 = 0.3635819
     a1 = 0.4891775
     a2 = 0.1365995
     a3 = 0.0106411
     
     blackman = np.zeros((N, nventanas))
     nn = np.linspace(0, N-1, N)
     blackamanNuttall = a0 - a1 * np.cos((2 * np.pi * nn)/(N - 1)) + a2 * np.cos((4 * np.pi * nn)/(N - 1)) - a3 * np.cos((6 * np.pi * nn)/(N - 1)) 
     if nventanas > 1 :
          for ii in range(nventanas):
               blackman[:, ii] = blackamanNuttall
     else:
          return  blackamanNuttall
     return blackman     
