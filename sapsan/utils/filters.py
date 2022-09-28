import numpy as np
        
def spectral(im: np.ndarray, fm: int):
    from scipy import fftpack
    
    #Spectral Filter
    form = np.shape(im)
    axis = len(form)
    dim = form[0]    
    half = np.zeros(axis, dtype=int)
    
    assert axis in [2,3], "Input variable has to be in the following format: for 3D [D, H, W], and for 2D [D, H]"

    im_fft = fftpack.fftn(im)

    im_fft = np.fft.fftshift(im_fft)

    for i in range(axis):
        half[i] = form[i]/2

    temp = np.zeros(np.shape(im_fft), dtype=complex)
    if axis==2: temp[half[0]-fm:half[0]+fm,
                        half[1]-fm:half[1]+fm]=im_fft[half[0]-fm:half[0]+fm,
                                                                half[1]-fm:half[1]+fm]
    else:  temp[half[0]-fm:half[0]+fm,
                half[1]-fm:half[1]+fm,
                half[2]-fm:half[2]+fm]=im_fft[half[0]-fm:half[0]+fm,
                                                        half[1]-fm:half[1]+fm,
                                                        half[2]-fm:half[2]+fm]

    for i in range(half[0]-fm,half[0]+fm):
        for j in range(half[1]-fm,half[1]+fm):
            if axis==2:
                if np.sqrt((half[0]-i-0.5)**2+(half[1]-j-0.5)**2)>fm: temp[i,j]=0
            else:
                for k in range(half[2]-fm,half[2]+fm):
                    if np.sqrt((half[0]-i-0.5)**2+(half[1]-j-0.5)**2+(half[2]-k-0.5)**2)>fm: temp[i,j,k]=0

    im_fft = np.fft.ifftshift(temp)
    im_new = fftpack.ifftn(im_fft).real

    return im_new

def box(im: np.ndarray, ksize):
    import cv2
    #Box filter

    form = np.shape(im)
    axis = len(form)
    dim = form[0]
        
    assert axis in [2], "Box filter only supports 2D image input of shape [D, H]"
        
    im_new = cv2.boxFilter(im, ddepth=-1, ksize=(int(ksize), int(ksize)))

    return im_new


def gaussian(im: np.ndarray, sigma):
    from scipy import ndimage        
    
    #Gaussian Filter
    return ndimage.gaussian_filter(im, sigma)

