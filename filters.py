import time
from scipy import fftpack
import numpy as np

class Filters:
    #def __init__(self):
    #    self.filt = getattr(self, self.filtname)

    def boxfilt(self, im):
        #Box filter
        tgr = time.time()
        im_new = cv2.boxFilter(im, ddepth=-1,ksize=(self.fm,self.fm))
        print('Applied box filter in: %.1f s'%(time.time()-tgr))
        return im_new

    
    def spectral(self, im, dim):
        #Spectral Filter        
        tgr = time.time()
        im_fft = fftpack.fftn(im).real
        
        im_fft = np.fft.fftshift(im_fft)

        half = int(dim/2)
        
        tgr = time.time()
        temp = np.zeros(np.shape(im_fft))
        if self.axis==2: temp[half-self.fm:half+self.fm,
                              half-self.fm:half+self.fm]=im_fft[half-self.fm:half+self.fm,
                                                                half-self.fm:half+self.fm]
        else: temp[half-self.fm:half+self.fm,
                   half-self.fm:half+self.fm,
                   half-self.fm:half+self.fm]=im_fft[half-self.fm:half+self.fm,
                                                    half-self.fm:half+self.fm,
                                                    half-self.fm:half+self.fm]
        
        for i in range(half-self.fm,half+self.fm):
            for j in range(half-self.fm,half+self.fm):
                if self.axis==2:
                    if np.sqrt((half-i-0.5)**2+(half-j-0.5)**2)>self.fm: temp[i,j]=0
                else:
                    for k in range(half-self.fm,half+self.fm):
                        if np.sqrt((half-i-0.5)**2+(half-j-0.5)**2+(half-k-0.5)**2)>self.fm: temp[i,j,k]=0
        
        im_fft = np.fft.ifftshift(temp)
        im_new = fftpack.ifftn(im_fft).real
        print('Applied spectral filter in: %.1f s'%(time.time()-tgr))

        return im_new    
