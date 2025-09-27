import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.ndimage import rotate, zoom
import skimage as sk
import skimage.io as skio



def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

def get_points_notebook(im1, im2):
    points = []
    
    def onclick(event):
        if event.inaxes is not None:
            points.append([event.xdata, event.ydata])
            ax.plot(event.xdata, event.ydata, 'ro', markersize=8)
            fig.canvas.draw()
            if len(points) == 2:
                plt.close()
    
    # First image
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(im1)
    ax.set_title('Click 2 points on first image')
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    p1, p2 = points[:2]
    points = []
    
    # Second image
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(im2)
    ax.set_title('Click 2 points on second image')
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    p3, p4 = points[:2]
    return (p1, p2, p3, p4)

def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int) (np.abs(2*r+1 - R))
    cpad = (int) (np.abs(2*c+1 - C))
    return np.pad(
        im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
             (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
             (0, 0)], 'constant')

def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape
    
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    if dscale < 1:
        im1 = zoom(im1, (dscale, dscale, 1), order=1)
    else:
        im2 = zoom(im2, (1./dscale, 1./dscale, 1), order=1)
    return im1, im2

def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = rotate(im1, dtheta*180/np.pi, axes=(0,1), reshape=True, order=1)
    return im1, dtheta

def match_img_size(im1, im2):
    # Make images the same size
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    if h1 < h2:
        im2 = im2[int(np.floor((h2-h1)/2.)) : -int(np.ceil((h2-h1)/2.)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1-h2)/2.)) : -int(np.ceil((h1-h2)/2.)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2-w1)/2.)) : -int(np.ceil((w2-w1)/2.)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1-w2)/2.)) : -int(np.ceil((w1-w2)/2.)), :]
    assert im1.shape == im2.shape
    return im1, im2

def align_images(im1, im2, mode = 'default'):
    if mode == 'default':
        pts = get_points(im1, im2)
    elif mode == "notebook":
        pts = get_points_notebook(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2


if __name__ == "__main__":
    # 1. load the image
    # 2. align the two images by calling align_images
    # Now you are ready to write your own code for creating hybrid images!
    
    #------------------------------------------------------------
    #align kenta and tsumi
    
    campanile = sk.img_as_float(skio.imread('../images/campanile.jpg'))
    venice = sk.img_as_float(skio.imread('../images/venice.jpg'))
    campanile_align, venice_align = align_images(campanile, venice)
    plt.imsave("campanile_aligned.png", campanile_align)
    plt.imsave("venice_aligned.png", venice_align)
    
    #------------------------------------------------------------
    #align chris and sriram
    # chris = skio.imread('../images/chris.jpg')
    # chris = np.rot90(chris, k=-1)
    # #chris = sk.img_as_float(chris)
    # sriram = skio.imread('../images/sriram.jpg')
    # sriram = np.rot90(sriram, k=-1) 
    # #sriram = sk.img_as_float(sriram)

    # chris_align, sriram_align = align_images(chris, sriram)
    # skio.imsave("chris_aligned.jpg", chris_align)
    # skio.imsave("sriram_aligned.jpg", sriram_align)
    
    #------------------------------------------------------------
    #align derek and cat
    # derek = skio.imread('./DerekPicture.jpg')
    # nutmeg = skio.imread("./nutmeg.jpg")
    
    # derek_align, nutmeg_align = align_images(derek, nutmeg)
    # skio.imsave("derek_aligned.jpg", derek_align)
    # skio.imsave("nutmeg_aligned.jpg", nutmeg_align)