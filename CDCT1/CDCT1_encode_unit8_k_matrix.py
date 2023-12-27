
import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy import r_
#from matplotlib.pyplot import imread
#from matplotlib.pyplot import imsave
import scipy.fftpack
import imageio
import os, os.path
import cv2

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def AjEd8bitEncode_K_matrix(im , hiddenim, img_no):

    imsize = im.shape
    width = int(im.shape[1])
    height = int(im.shape[0])
    dct = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            dct[i:(i+8),j:(j+8)] = dct2( im[i:(i+8),j:(j+8)] )
    newhidden = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            newhidden[i:(i+8),j:(j+8)] = dct2(hiddenim[i:(i+8),j:(j+8)])

    dct_stego = dct
    index = 2

    
    alpha = np.array([[2,1,1,1,1,1],      # Modified from 8x8 QUANTIZATION TABLE
                        [1,0.1,0.1,0.1,0.1,0.1],    
                        [1,0.1,0.1,0.1,0.1,0.1],
                        [1,0.1,0.1,0.1,0.1,0.1],
                        [1,0.1,0.1,0.1,0.1,0.1],
                        [1,0.1,0.1,0.1,0.1,0.1]])
    
    alpha3D = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    beta = 25
    scale = alpha3D*beta
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            dct_stego[(i+index):(i+8),(j+index):(j+8)] = np.divide(newhidden[i:(i+8-index),j:(j+8-index)],scale,where=scale!=0)
             
    im_dct = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            im_dct[i:(i+8),j:(j+8)] = idct2( dct_stego[i:(i+8),j:(j+8)] )
    
    im_dct[im_dct < 0] = 0
    im_dct[im_dct > 1] = 1

    if img_no < 10:
        imageio.imwrite(Output_StegoPath + 'stego_image_00' + str(img_no) +'.png', im_dct)
        
    elif img_no < 100 and img_no > 9:
        imageio.imwrite(Output_StegoPath + 'stego_image_0' + str(img_no) +'.png', im_dct)
        
    else:
        imageio.imwrite(Output_StegoPath + 'stego_image_' + str(img_no) +'.png', im_dct)
    
#--------------------------------------------------------
Cover_path  = "path to load cover images";
Hidden_path = "path to load hidden images";
Output_StegoPath = "path to save stego images";
image_Cover = []
image_Hidden = []
image_Stego = []

coverimg =0;
hiddenimg =0;
#---------------------------------------------------------------------------------------------------
# Load Image
#---------------------------------------------------------------------------------------------------

valid_images = [".png"]
for fc in os.listdir(Cover_path):
    ext = os.path.splitext(fc)[1]
    if ext.lower() not in valid_images:
        continue
    im1 = imageio.v2.imread(os.path.join(Cover_path,fc))
    C_image = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
    image_Cover.append(C_image/255)
    coverimg = coverimg +1;
  
for fh in os.listdir(Hidden_path):
    ext = os.path.splitext(fh)[1]
    if ext.lower() not in valid_images:
        continue
    H_image = imageio.v2.imread(os.path.join(Hidden_path,fh)).astype(float)/255
    H_image = H_image[:,:,0:3]
    image_Hidden.append(H_image)
    hiddenimg = hiddenimg +1;

#---------------------------------------------------------------------------------------------------
# Main Process
#--------------------------------------------------------------------------------------------------- 
for n in range(0,coverimg): 
    print('Process image no :',n)
    im_o = image_Cover[n]
    h_im_o = image_Hidden[n]
    AjEd8bitEncode_K_matrix(im_o , h_im_o,n)
