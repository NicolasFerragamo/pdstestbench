#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:04:10 2019

@author: nico
"""

import os
import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import control
from numpy import pi, diff, unwrap, angle
os.system ("clear") # limpia la terminal de python
plt.close("all")    #cierra todos los graficos 

num = np.array([1,0, 1]) #numerador de  H[b2, b1, b0]
den = np.array([1, 0, 0]) #denumerador de  H[a2, a1, a0]

z, p, k = sig.tf2zpk(num,den)
print("Z =", z, "\n", "P =", p, "\n", "K =", k, "\n")


ww, hh = sig.freqz(num, den)
ww = ww / np.pi
group_delay = -diff(unwrap(angle(hh)))/diff(ww)
eps = np.finfo(float).eps


plt.figure("Filtro FIR") 
ax1 = plt.subplot(2, 1, 1)
ax1.set_title('Módulo')
ax1.plot(ww, 20 * np.log10(abs(hh+eps)), label= "respuesta de Modulo")
#ax1.plot(ww[:-1], group_delay, label= "retardo de grupo")
ax1.set_xlabel('Frequencia normalizada')
ax1.set_ylabel('Modulo [dB]')
plt.grid()
ax2 = plt.subplot(2, 1, 2)
ax2.set_title('Fase')
ax2.plot(ww, np.angle(hh), label= "respuesta de fase")
ax2.set_xlabel('Frequencia normalizada')
ax2.set_ylabel('[Rad]')
plt.grid()
plt.legend()
plt.show()
plt.tight_layout()



tf = control.TransferFunction(num,den,1)
print (tf)
control.pzmap(tf, Plot=True, title='Pole Zero Map', grid=True)

p = control.pole(tf)
print(p)