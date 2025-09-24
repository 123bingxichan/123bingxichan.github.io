import numpy as np
import skimage as sk
import skimage.io as skio
import scipy
import sys
import os

from convolve import convolve4l, convolve2l

def save_output_im(im_out, fname):  
    """Saves an image and outputs it

    Inputs:
        im_out: Stacked image in (H x W x 3) (r g b)
        fname name of file
    """
    # save the image
    fname = f'../images/{fname}.jpg'
    if im_out.dtype == np.float64:
        im_out = (im_out * 255).astype(np.uint8)
    else:
        raise ValueError(f"Invalid image type {im_out.dtype}")
    skio.imsave(fname,im_out)
    
    
file = '../images/cameraman.png'
im = skio.imread(file)
im = sk.img_as_float(im) 
im = sk.color.rgba2rgb(im)
im = sk.color.rgb2gray(im)  # Convert to grayscale


boxfilt = np.full((15,15), 1/(15**2))
Dx = np.array([1,0,-1]).reshape(1,3)
Dy = np.array([
    [1],
    [0],
    [-1]
    ])

print(im.shape, Dx.shape)
#save_output_im(convolve2l(im, Dx), '../images/convolved_Dx_cameraman')
save_output_im(scipy.signal.convolve2d(im,Dx), '../images/scipy_Dx_cameraman')
save_output_im(scipy.signal.convolve2d(im,Dy), '../images/scipy_Dy_cameraman')

