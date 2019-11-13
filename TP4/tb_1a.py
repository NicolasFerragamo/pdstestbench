#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:07:23 2019

@author: nico
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import control

num = np.array([1/3, 1/3, 1/3])
den = np.array([1, 0, 0])

z, p, k = sig.tf2zpk(num,den)

ww, hh = sig.freqz(num, den)
ww = ww / np.pi

#plt.figure("Filtro FIR") 
#plt.subplot(2, 1, 1)
#plt.title('Módulo')
#plt.plot(ww, 20 * np.log10(abs(hh)))
#plt.xlabel('Frequencia normalizada')
#plt.ylabel('Modulo [dB]')
#plt.grid()
#plt.subplot(2, 1, 2)
#plt.title('Fase')
#plt.plot(ww, np.angle(hh))
#plt.xlabel('Frequencia normalizada')
#plt.ylabel('[Rad]')
#plt.grid()
#plt.show()
#plt.tight_layout()

plt.figure("Filtro FIR") 
ax1 = plt.subplot(2, 1, 1)
ax1.set_title('Módulo')
ax1.plot(ww, 20 * np.log10(abs(hh)))
ax1.set_xlabel('Frequencia normalizada')
ax1.set_ylabel('Modulo [dB]')
plt.grid()
ax2 = plt.subplot(2, 1, 2)
ax2.set_title('Fase')
ax2.plot(ww, np.angle(hh))
ax2.set_xlabel('Frequencia normalizada')
ax2.set_ylabel('[Rad]')
plt.grid()
plt.show()
plt.tight_layout()



tf = control.TransferFunction(num,den)
print (tf)
control.pzmap(tf, Plot=True, title='Pole Zero Map', grid=True)

#plt.figure("Diagramas de polos y ceros")
#plt.plot(z.real, z.imag, 'o')
#plt.plot(p.real, p.imag, 'x')
#plt.xlabel('Re')
#plt.ylabel('Im')
#plt.grid()
#plt.show()