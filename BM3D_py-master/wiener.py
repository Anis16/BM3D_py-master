import os
import cv2
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian, convolve2d
from psnr import compute_psnr
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s = img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return dummy
def gaussian_kernel(kernel_size = 3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h

kernel=gaussian_kernel(3)
im_dir = 'test_data/image'
sigma = 10
noisy_dir = 'test_data/sigma' + str(sigma)
for im_name in os.listdir(im_dir):
    A=np.zeros(4)

    im_path = os.path.join(im_dir, im_name)
    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    noisy_im_path = os.path.join(noisy_dir, im_name)
    noisy_im = cv2.imread(noisy_im_path, cv2.IMREAD_GRAYSCALE)
    im_filtered=wiener_filter(noisy_im, kernel, K=10)
    A[0]=compute_psnr(noisy_im,im)
    A[1]=mean_squared_error(noisy_im,im)
    A[2]=compute_psnr(im_filtered, im)
    A[3]=mean_squared_error(im_filtered,im)
    print(A)
