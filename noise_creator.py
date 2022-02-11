"""
File: noise_creator.py
Author: Abel John Oakley
Program: Takes an image as input and applies a gaussian noise
onto it. This is to create adversial images. 
"""

import numpy as np
import os
import cv2

def noisy(image):
    row,col,ch= image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss

    return noisy