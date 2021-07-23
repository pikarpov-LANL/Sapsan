import numpy as np
        
def spectral(im: np.ndarray, fm: int):
    from scipy import fftpack
    
    #Spectral Filter
    axis = len(np.shape(im))
    dim = np.shape(im)[0]
    form = np.shape(im)
    half = np.zeros(len(form))

    im_fft = fftpack.fftn(im)

    im_fft = np.fft.fftshift(im_fft)

    half[0] = int(form[0]/2)
    half[1] = int(form[1]/2)
    if axis==3: half[2] = int(form[2]/2)
    half = [int(i) for i in half]

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

    dim = np.shape(im)[0]
    axis = len(np.shape(im))

    if axis==2: im_new = cv2.boxFilter(im, ddepth=-1,ksize=ksize)
    else: raise ValueError('Box filter only supports 2D image input')

    return im_new


def gaussian(im: np.ndarray, sigma):
    from scipy import ndimage
    
    #Gaussian Filter
    return ndimage.gaussian_filter(im, sigma)

