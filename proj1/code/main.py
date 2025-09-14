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
    lowest_rows = row_sums[0:35, 1].astype(int)  # Column 1 contains the row indices

    # # Modify the original image for debugging
    # for i, row_idx in enumerate(lowest_rows):
    #     if i % 2 == 0:  # Even index (0, 2, 4) - make white
    #         im[row_idx, ::2] = 1  # Every other column (even indices) = white
    #     else:  # Odd index (1, 3) - make black  
    #         im[row_idx, 1::2] = 0  # Every other column (odd indices) = black
    
    #group the rows into 2 borders (middle of image)
    h_borders = np.sort(lowest_rows)
    borders = []
    tol = 200
    prev = []
    for h in h_borders:
        prev.append(h)
        if len(prev) > 1 and h - prev[-2] > tol:
            borders.append(np.floor(np.mean(prev[:-1])).astype(int))
            prev = prev[-1:]
    if len(prev) > 0:
        borders.append(np.floor(np.mean(prev)).astype(int))
    
    #found two borders! Split into 3 sections
    if len(borders) != 2:
        print("No two borders found, returning naive_align")
        return naive_align(im)
    b = im[:borders[0]]
    g = im[borders[0]:borders[1]]
    r = im[borders[1]:]
    
    
    # Make all their x the same as the min
    min_x = min(b.shape[0], g.shape[0], r.shape[0])
    orig_x = im.shape[0] // 3  # original channel height (approx)
    # If more than an 8th of the image is lost, return naive_align
    if orig_x - min_x > orig_x // 6:
        print("Too much image lost, returning naive_align")
        return naive_align(im)
    b, g, r = b[:min_x, :], g[:min_x, :], r[:min_x, :]
    
    im_out = np.dstack([r, g, b])
    return im_out        

def std_border_rem(im, tol):
    """
    Remove border regions by cropping rows/cols that deviate from mean std
    
    Args:
        im: input image
        tol: tolerance for standard deviation threshold
    
    Returns:
        cropped image with borders removed
    """
    # Calculate standard deviation for each column and row
    col_stds = np.std(im, axis=0)  # std for each column
    row_stds = np.std(im, axis=1)  # std for each row
    
    col_mean = np.mean(col_stds)
    row_mean = np.mean(row_stds)
    
    cols_to_keep = []
    for ci, c_std in enumerate(col_stds):
        if not ((c_std > col_mean + tol) or (c_std < col_mean - tol)):
            cols_to_keep.append(ci)
            
    rows_to_keep = []
    for ri, r_std in enumerate(row_stds):
        if not ((r_std > row_mean + tol) or (r_std < row_mean - tol)):
            rows_to_keep.append(ri)
    
    return im[np.array(rows_to_keep), :][:, np.array(cols_to_keep)]

def prcnt_border_rem(im, percentile=5):
    """
    Alternative border removal using percentile-based cropping
    
    Args:
        im: input image
        percentile: percentage of rows/cols to remove from each side
    
    Returns:
        cropped image
    """
    height, width = im.shape[:2]
    
    # Calculate how many pixels to remove from each side
    rows_to_remove = int(height * percentile / 100)
    cols_to_remove = int(width * percentile / 100)
    
    # Crop symmetrically
    if len(im.shape) == 3:
        return im[rows_to_remove:height-rows_to_remove, 
                 cols_to_remove:width-cols_to_remove, :]
    else:
        return im[rows_to_remove:height-rows_to_remove, 
                 cols_to_remove:width-cols_to_remove]
        
def l2_align(r_channel, g_channel, b_channel,rndx, rdx, rndy, rdy, gndx, gdx, gndy, gdy):
    """
    Align R and G channels to B using L2 norm minimization.
    B is always the reference channel.
    """
    # aligned_im = cropf(im)

    # r_channel = aligned_im[:, :, 0]
    # g_channel = aligned_im[:, :, 1]
    # b_channel = aligned_im[:, :, 2]

    best_r = (np.inf, 0, 0, r_channel)  # (loss, dx, dy, shifted_crop)
    best_g = (np.inf, 0, 0, g_channel)

    for dx in range(rndx, rdx + 1):
        for dy in range(rndy, rdy + 1):
            # shift by (dy, dx)
            r_shift = np.roll(r_channel, shift=(dy, dx), axis=(0, 1))

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
            b_crop = b_channel[y_slice, x_slice]

            # compute L2 loss normalized by pixels
            # r_loss = np.linalg.norm(b_crop - r_crop) / b_crop.size
            
            # Use negative normalized cross-correlation (NCC) as the loss (higher NCC is better, so we negate it for minimization)
            r_ncc = np.sum((b_crop - b_crop.mean()) * (r_crop - r_crop.mean()))
            r_ncc /= (np.sqrt(np.sum((b_crop - b_crop.mean())**2)) * np.sqrt(np.sum((r_crop - r_crop.mean())**2)) + 1e-8)
            r_loss = -r_ncc
            if r_loss < best_r[0]:
                best_r = (r_loss, dx, dy, r_crop.copy())

    
    for dx in range(gndx, gdx + 1):
        for dy in range(gndy, gdy + 1):
            # shift by (dy, dx)
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

            g_crop = g_shift[y_slice, x_slice]
            b_crop = b_channel[y_slice, x_slice]

            # compute L2 loss normalized by pixels
            #g_loss = np.linalg.norm(b_crop - g_crop) / b_crop.size
            g_ncc = np.sum((b_crop - b_crop.mean()) * (g_crop - g_crop.mean()))
            g_ncc /= (np.sqrt(np.sum((b_crop - b_crop.mean())**2)) * np.sqrt(np.sum((g_crop - g_crop.mean())**2)) + 1e-8)
            g_loss = -g_ncc

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

    # print("Best R shift:", best_r[1:3], "Loss:", best_r[0])
    # print("Best G shift:", best_g[1:3], "Loss:", best_g[0])

    return im_out,  best_r[1:3], best_g[1:3]

def gauss_filt(im, s):
    kernel_size = int(6 * s + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    x = np.arange(kernel_size) - kernel_size // 2
    kernel_1d = np.exp(-(x**2) / (2 * s**2))
    kernel_1d = kernel_1d / np.sum(kernel_1d)
    
    if len(im.shape) == 3:
        filtered = np.zeros_like(im)
        for i in range(im.shape[2]):
            temp = scipy.ndimage.convolve1d(im[:, :, i], kernel_1d, axis=1, mode='constant')
            filtered[:, :, i] = scipy.ndimage.convolve1d(temp, kernel_1d, axis=0, mode='constant')
    else:
        temp = scipy.ndimage.convolve1d(im, kernel_1d, axis=1, mode='constant')
        filtered = scipy.ndimage.convolve1d(temp, kernel_1d, axis=0, mode='constant')
    
    mean_val = np.mean(filtered)
    std_val = np.std(filtered)
    
    return filtered, mean_val, std_val

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
       filter = gauss_filt
    # elif filter_type == 'mean':
    #     filter= mean_filt
    else:
        raise ValueError(f"Invalid filter type {filter_type}")

    if k == 1:
        return [im[::2,::2]]
    #filter image w filter
    im, mean_val, std_val = filter(im, s)
    #print(f"Gaussian filter - Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        
    #begin subsampling
    subsample = im[::2,::2]
    return [subsample] + pyramid(subsample, k-1, s, filter_type)

def pyramid_align(im, k, s, cropf, filter_type='gauss'):
    """
    Align the image using the pyramid
    
    tol is the amount of search from lower image to upper image
    """
    im = border_align(prcnt_border_rem(im))
    
    # Extract individual color channels
    r = im[:, :, 0]
    g = im[:, :, 1] 
    b = im[:, :, 2]
    
    # Create pyramids for each channel separately
    r_pyramid = pyramid(r, k, s, filter_type)[::-1]  # start with smallest image
    g_pyramid = pyramid(g, k, s, filter_type)[::-1]
    b_pyramid = pyramid(b, k, s, filter_type)[::-1]
    
    rdx, rdy, gdx, gdy = 0, 0, 0, 0  # set initial search of smallest image to be 5
    offset = 20
    
    # Process each level of the pyramid
    for i in range(len(r_pyramid)):
        r_level = r_pyramid[i]
        g_level = g_pyramid[i]
        b_level = b_pyramid[i]
        
        aligned_im, best_r, best_g = l2_align(r_level, g_level, b_level,  
                                              2* rdx - offset,
                                              2* rdx + offset,
                                              2* rdy - offset,
                                              2* rdy + offset,
                                              2* gdx - offset,
                                              2* gdx + offset,
                                              2* gdy - offset,
                                              2* gdy + offset,
                                              )
        rdx, rdy, gdx, gdy = best_r[0], best_r[1], best_g[0], best_g[1]
        #offset = int(np.ceil(1.5 * offset))
    
    print("best r and g vectors: ", best_r, best_g)
    return aligned_im, best_r, best_g

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
    # skio.imshow(fname)
    # skio.show()

file = '../images/harvesters.tif'
im = skio.imread(file)
im = sk.img_as_float(im)

#try naive:
#save_output_im(naive_align(im), 'naive_stack')

#try border:
# save_output_im(border_align(im), 'border_stack')

#try l2
# save_output_im(l2_align(im,30,30)[0], 'l2_try22')


#try pyramid:
# save_output_im(pyramid_align(im,5,3, naive_align)[0], 'pyramid_try1')

#apply to all files in the selected folder:
selected_dir = '../images/final/selected'
for file in os.listdir(selected_dir):
    if not (file.lower().endswith('.jpg') or file.lower().endswith('.tif')):
        continue
    if file.endswith('.jpg'):
        k = 1
    else:
        k = 4
    im = skio.imread(f'{selected_dir}/{file}')
    im = sk.img_as_float(im)
    print('Now aligning: ', file)
    save_output_im(pyramid_align(im,k,3, naive_align)[0],f'../images/final/full_run2/pyramid_{file}')

#Verify pyramid
# p = pyramid(im,7,5)[3:]
# for i in range(len(p)):
#     save_output_im(p[i], f'../images/emir_{i}')

#save_output_im(border_align(prcnt_border_rem(im)), 'border_aligned_image')
