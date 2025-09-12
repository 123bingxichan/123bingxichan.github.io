import numpy as np
import scipy.ndimage
import skimage as sk
import skimage.io as skio
import os

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

def l2_align(im, max_disp_x, max_disp_y):
    """
    Align R and G channels to B using L2 norm minimization.
    B is always the reference channel.
    """
    aligned_im = border_align(im)

    r_channel = aligned_im[:, :, 0]
    g_channel = aligned_im[:, :, 1]
    b_channel = aligned_im[:, :, 2]

    best_r = (np.inf, 0, 0, r_channel)  # (loss, dx, dy, shifted_crop)
    best_g = (np.inf, 0, 0, g_channel)

    for dx in range(-max_disp_x, max_disp_x + 1):
        for dy in range(-max_disp_y, max_disp_y + 1):
            # shift by (dy, dx)
            r_shift = np.roll(r_channel, shift=(dy, dx), axis=(0, 1))
            g_shift = np.roll(g_channel, shift=(dy, dx), axis=(0, 1))

            # crop overlapping region between shifted and reference (b)
            if dx >= 0:
                x_slice = slice(dx, None)
            else:
                x_slice = slice(0, dx)
            if dy >= 0:
                y_slice = slice(dy, None)
            else:
                y_slice = slice(0, dy)

            r_crop = r_shift[y_slice, x_slice]
            g_crop = g_shift[y_slice, x_slice]
            b_crop = b_channel[y_slice, x_slice]

            # compute L2 loss normalized by pixels
            r_loss = np.linalg.norm(b_crop - r_crop) / b_crop.size
            g_loss = np.linalg.norm(b_crop - g_crop) / b_crop.size

            if r_loss < best_r[0]:
                best_r = (r_loss, dx, dy, r_crop.copy())
            if g_loss < best_g[0]:
                best_g = (g_loss, dx, dy, g_crop.copy())

    # crop all to smallest common size
    min_height = min(best_r[3].shape[0], best_g[3].shape[0], b_channel.shape[0])
    min_width  = min(best_r[3].shape[1], best_g[3].shape[1], b_channel.shape[1])

    im_out = np.dstack([
        best_r[3][:min_height, :min_width],
        best_g[3][:min_height, :min_width],
        b_channel[:min_height, :min_width]
    ])

    print("Best R shift:", best_r[1:3], "Loss:", best_r[0])
    print("Best G shift:", best_g[1:3], "Loss:", best_g[0])

    return im_out

def pyramid(im, k,s ,filter_type='gauss'):
    """
    Create a pyramid of images of im with a kernel of size s x s

    Args:
        im: image to create pyramid of
        k: number of levels in the pyramid
        s: size of the kernel
        filter_type: type of filter to use

    Outputs:
        pyramid: list of images
    """
    if filter_type == 'gauss':
       filter= scipy.ndimage.gaussian_filter
    # elif filter_type == 'mean':
    #     filter= scipy.ndimage.uniform_filter
    else:
        raise ValueError(f"Invalid filter type {filter_type}")

    if k == 1:
        return [im[::2,::2]]
    #filter image w filter
    im = filter(im, sigma=s)
        
    #begin subsampling
    subsample = im[::2,::2]
    return [subsample] + pyramid(subsample, k-1, s, filter_type)

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
# save_output_im(border_align(im), 'border_stack')

#try l2
# save_output_im(l2_align(im,30,30), 'l2_try22')

#apply to all files in the images folder:
for file in os.listdir('../images'):
    if file.endswith('.jpg'):
        im = skio.imread(f'../images/{file}')
        im = sk.img_as_float(im)
        save_output_im(l2_align(im,30,30), f'../images/final/l2_{file}')