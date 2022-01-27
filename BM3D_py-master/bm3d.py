from utils import add_gaussian_noise, symetrize
from bm3d_1st_step import bm3d_1st_step
from bm3d_2nd_step import bm3d_2nd_step
from psnr import compute_psnr
from sklearn.metrics import mean_squared_error


def run_bm3d(noisy_im, sigma,
             n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H,
             n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W):
    k_H = 8 if (tau_2D_H == 'BIOR' or sigma < 40.) else 12
    k_W = 8 if (tau_2D_W == 'BIOR' or sigma < 40.) else 12

    noisy_im_p = symetrize(noisy_im, n_H)
    img_basic = bm3d_1st_step(sigma, noisy_im_p, n_H, k_H, N_H, p_H, lambda3D_H, tauMatch_H, useSD_H, tau_2D_H)
    img_basic = img_basic[n_H: -n_H, n_H: -n_H]

    assert not np.any(np.isnan(img_basic))
    img_basic_p = symetrize(img_basic, n_W)
    noisy_im_p = symetrize(noisy_im, n_W)
    img_denoised = bm3d_2nd_step(sigma, noisy_im_p, img_basic_p, n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W)
    img_denoised = img_denoised[n_W: -n_W, n_W: -n_W]

    return img_basic, img_denoised


if __name__ == '__main__':
    import os
    import cv2
    import numpy as np

    # <hyper parameter> -------------------------------------------------------------------------------
    n_H = 16
    k_H = 8
    N_H = 16
    p_H = 3
    lambda3D_H = 2.7  # ! Threshold for Hard Thresholding
    useSD_H = False
    tau_2D_H = 'BIOR'

    n_W = 16
    k_W = 8
    N_W = 32
    p_W = 3
    useSD_W = True
    tau_2D_W = 'DCT'
    # <\ hyper parameter> -----------------------------------------------------------------------------

    im_dir = 'test_data/image'
    save_dir = 'temp_test_result'
    os.makedirs(save_dir, exist_ok=True)
    i=0
    j=0
    for im_name in os.listdir(im_dir):
        i+=1
        sigma_list = [2, 5, 10, 20, 30, 40, 60, 80, 100]
        A=np.zeros((len(os.listdir(im_dir)),4))
        sigma=5
        print(im_name, '  ', sigma)
        tauMatch_H = 2500 if sigma < 35 else 5000  # ! threshold determinates similarity between patches
        tauMatch_W = 400 if sigma < 35 else 3500  # ! threshold determinates similarity between patches
        noisy_dir = 'test_data/sigma' + str(sigma)

        im_path = os.path.join(im_dir, im_name)
        im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        noisy_im_path = os.path.join(noisy_dir, im_name)
        noisy_im = cv2.imread(noisy_im_path, cv2.IMREAD_GRAYSCALE)

        im1, im2 = run_bm3d(noisy_im, sigma,
                                n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H,
                                n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W)

        psnr_1st = compute_psnr(im, im1)
        psnr_2nd = compute_psnr(im, im2)
        eqm_1st=mean_squared_error(im,im1)
        eqm_2nd=mean_squared_error(im,im2)
        A[i]=[psnr_1st,psnr_2nd,eqm_1st,eqm_2nd]
    print(A)