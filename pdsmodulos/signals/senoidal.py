#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:32:09 2019

@author: nico
"""


import numpy as np



def seno (fs=1000, f0=1, N=1000, a0=2, p0=0) :
    w = 2 * np.pi * f0
    tt = np.arange(0, (N - 1)/fs, 1/fs)  # ¿cual es la diferencia con linespace?
    signal = a0  * np.sin(w * tt + p0)
    return tt, signal
