# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio

# name of the input file
imname = '../images/cathedral.jpg'

# read in the image
im = skio.imread(imname)

# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)

def naive_align(im):
    
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(np.uint32)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    im_out = np.dstack([r, g, b])
    return im_out

def border_align(im):
    """
    Attempt to find the borders between the BGR channels and align the image accordingly
    First, try to sort the rows by their lowest sum, 
    """
    #create an array of (row-wise sum, h-index)
    row_sums = np.sum(im, axis=1)
    sum_heights = np.arange(im.shape[0])
    row_sums = np.column_stack((row_sums, sum_heights))
    row_sums = row_sums[row_sums[:, 0].argsort()]
    #now we have the rows sorted by their lowest sum
    
    # Get the row indices of the 15 lowest sum rows
    lowest_rows = row_sums[0:15, 1].astype(int)  # Column 1 contains the row indices

    # # Modify the original image for debugging
    # for i, row_idx in enumerate(lowest_rows):
    #     if i % 2 == 0:  # Even index (0, 2, 4) - make white
    #         im[row_idx, ::2] = 1  # Every other column (even indices) = white
    #     else:  # Odd index (1, 3) - make black  
    #         im[row_idx, 1::2] = 0  # Every other column (odd indices) = black
    
    #group the rows into 4
    h_borders = np.sort(lowest_rows)
    borders = []
    tol = 10
    prev = []
    for h in h_borders:
        prev.append(h)
        if len(prev) > 1 and h - prev[-2] > tol:
            borders.append(np.floor(np.mean(prev[:-1])).astype(int))
            prev = prev[-1:]
    if len(prev) > 0:
        borders.append(np.floor(np.mean(prev)).astype(int))
    
    #found four borders!
    # print(h_borders)
    # print(borders)
    b = im[borders[0]:borders[1]]
    g = im[borders[1]:borders[2]]
    r = im[borders[2]:borders[3]]
    
    #make all their x the same as the min
    min_x = min(b.shape[0], g.shape[0], r.shape[0])
    b,g,r = b[:min_x,:], g[:min_x,:], r[:min_x,:]
    
    im_out = np.dstack([r, g, b])
    return im_out        
    
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
    
    # display the image
    skio.imshow(fname)
    skio.show()

#try naive:
#save_output_im(naive_align(im), 'naive_stack')

#try border:
save_output_im(border_align(im), 'border_stack')
