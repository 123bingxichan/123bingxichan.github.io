import numpy as np

#four for loops
def convolve4l(in1 , filt):
    """Try to copy scipy convolve2d

    Args:
        in1 (np.2darray): 2D array to be convolved on
        filter (np.2darray): array to pass in as the filter 
    """
    
    if filt.ndim == 1:
        filt = filt.reshape(1, -1) if len(filt) > 1 else filt.reshape(1, 1)
    filt = np.flip(filt, axis=(0,1))
    cx = filt.shape[0] // 2 
    cy = filt.shape[1] // 2 
    convolved = np.zeros(in1.shape)
    padded = np.zeros((in1.shape[0] + 2*cx, in1.shape[1] + 2*cy), dtype = float)
    padded[cx:-cx, cy:-cy] = in1
    
    if cx == 0 and cy == 0:
        padded = in1
    elif cx == 0:
        padded[:, cy:-cy] = in1
    elif cy == 0:
        padded[cx:-cx, :] = in1
    else:
        padded[cx:-cx, cy:-cy] = in1
    #convolve entire image
    for r in range(padded.shape[0] - (filt.shape[0]-1)):
        for c in range(padded.shape[1] - (filt.shape[1]-1)):
            #set the item at the correct index to result of using the filter function
            filtered = 0
            for i in range(filt.shape[0]):
                for j in range(filt.shape[1]):
                    filtered += filt[i,j] * padded[r + i, j + c]
            convolved[r,c] = filtered
    return convolved

#two for loops
def convolve2l(in1 , filt):
    """Try to copy scipy convolve2d

    Args:
        in1 (np.2darray): 2D array to be convolved on
        filter (np.2darray): array to pass in as the filter 
    """
    
    if filt.ndim == 1:
        filt = filt.reshape(1, -1) if len(filt) > 1 else filt.reshape(1, 1)
        
    filt = np.flip(filt, axis=(0,1))    
    cx = filt.shape[0] // 2
    cy = filt.shape[1] // 2
    convolved = np.zeros(in1.shape)
    padded = np.zeros((in1.shape[0] + 2*cx, in1.shape[1] + 2*cy), dtype = float)
    
    if cx == 0 and cy == 0:
        padded = in1
    elif cx == 0:
        padded[:, cy:-cy] = in1
    elif cy == 0:
        padded[cx:-cx, :] = in1
    else:
        padded[cx:-cx, cy:-cy] = in1
    
    #convolve entire image
    for r in range(padded.shape[0] - (filt.shape[0]-1)):
        for c in range(padded.shape[1] - (filt.shape[1]-1)):
            #set the item at the correct index to result of using the filter function
            convolved[r,c] = np.sum(np.multiply(filt,padded[r:r+filt.shape[0], c: c+filt.shape[1]]))

    return convolved

def test_convolve():
    # Simple 3x3 input matrix
    input_matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    
    # 3x3 identity filter (odd dimensions)
    input_filter = np.array([
        1,0,-1
    ])
    
    # Expected result: identity filter should return the same matrix
    expected = np.array([
        [5, 7, 9],
        [12, 15, 18],
        [11, 13, 15]
    ])
    
    got = convolve2l(input_matrix, input_filter)
    
    print("Input matrix:")
    print(input_matrix)
    print("\nInput filter:")
    print(input_filter)
    print("\nExpected:")
    print(expected)
    print("\nGot:")
    print(got)
    print("\nTest passed:", np.allclose(expected, got))
# # Run the test
# test_convolve()