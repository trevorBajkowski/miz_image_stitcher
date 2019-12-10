import cv2
from skimage import exposure
import numpy as np

def rgb_balance(im1, im2):
    matched = exposure.match_histograms(im1, im2, multichannel=True)
    return matched

def hsv_balance(im1, im2):
    im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2HSV)
    im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2HSV)
    im1[:,:,2] = exposure.match_histograms(im1[:,:,2], im2[:,:,2], multichannel=False)
    im1 = cv2.cvtColor(im1, cv2.COLOR_HSV2RGB)
    return im1

def sharpen(im, sigma=1.0, kernel_size=(5,5)):
    blurred = cv2.GaussianBlur(im, kernel_size, sigma)
    sharp = 1.5 * im - 1.5 * blurred
    sharp = np.max(sharp, np.zeros(sharp.shape))
    sharp = np.min(sharp, 255 * np.ones(sharp.shape))
    sharp = sharp.round().astype(np.uint8)
    low_contrast = np.absolute(im - blurred) < 5
    np.copyto(sharp, im, where=low_contrast)
    return sharp

def smooth(im, smoothing_alg="bilateral"):
    if smoothing_alg == "bilateral":
        smooth = cv2.bilateralFilter(im, 9, 60, 60)
    elif smoothing_alg == "median":
        smooth = cv2.medianBlur(im, 5)
    elif smoothing_alg == "gaussian":
        smooth = cv2.GaussianBlur(im, (5,5), 0)
    else:
        smooth = None
    return smooth
