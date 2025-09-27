import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage as sk
import skimage.io as skio
import scipy

def gl_stack(im, k, ):
    """Given an image, return a tuple of the gaussian and laplacian stack

    Args:
        im (ArrayLike): np.ndarray of shape [w, h, 3] or [w, h]
        k (int) : number of levels to the stack
    Output:
        tuple of first the gaussian stack, then the laplacian stack
        
    Example Usage:
        im_gs, im_ls = gl_stack(im)
    """
    KERNEL_SIZE = 15
    SIGMA = 2
    gaussian_2d = scipy.signal.windows.gaussian(KERNEL_SIZE, SIGMA)
    gaussian_2d = gaussian_2d.reshape(-1, 1) @ gaussian_2d.reshape(1, -1)
    gaussian_2d = gaussian_2d / np.sum(gaussian_2d)  

    gs = []
    ls = []
    
    
    for level in range(k):
        if level > 0:
            image = gs[level - 1] 
        else:
            #first sample,need to subtract by im
            image = im
            
        if len(im.shape) == 2: #grayscale
            gauss = scipy.signal.convolve2d(image, gaussian_2d, mode = 'same')
        elif len(im.shape) == 3: #rgb
            gauss = np.stack([scipy.signal.convolve2d(image[:,:,color], gaussian_2d, mode='same') for color in range(image.shape[2])], axis=2)
            
        gs.append(gauss)
        
        
        if level > 0:
            down = gs[level - 1] 
        else:
            #first sample,need to subtract by im
            down = im
            
        laplace = down - gs[level]
        ls.append(laplace)
    return gs, ls