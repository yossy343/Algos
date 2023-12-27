# Import functions and libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy import r_
import scipy.fftpack
import imageio
import os, os.path
import cv2


def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def AjEd8bit_k_matrix_DecodeProcess(imh,img_no):
    imsize = imh.shape
    dct_out = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            dct_out[i:(i+8),j:(j+8)] = dct2( imh[i:(i+8),j:(j+8)] )
            
    h8 = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            h8[i:(i+6),j:(j+6)] = dct_out[(i+2):(i+8),(j+2):(j+8)]

    reconst = np.zeros(imsize)
    alpha = np.array([[2,1,1,1,1,1,0,0],      # Modified from 8x8 QUANTIZATION TABLE
                        [1,0.1,0.1,0.1,0.1,0.1,0,0],    
                        [1,0.1,0.1,0.1,0.1,0.1,0,0],
                        [1,0.1,0.1,0.1,0.1,0.1,0,0],
                        [1,0.1,0.1,0.1,0.1,0.1,0,0],
                        [1,0.1,0.1,0.1,0.1,0.1,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0]])
    
    alpha3D = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    beta = 25
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            reconst[i:(i+8),j:(j+8)] = idct2( h8[i:(i+8),j:(j+8)]*alpha3D*beta)
    reconst[reconst < 0] = 0
    reconst[reconst > 1] = 1
       
    if img_no < 10:
       imageio.imwrite(Output_ReconstructPath + 'reconstruct_image_00' + str(img_no) +'.png', reconst)   
    elif img_no < 100 and img_no > 9:
       imageio.imwrite(Output_ReconstructPath + 'reconstruct_image_0' + str(img_no) +'.png', reconst)    
    else:
       imageio.imwrite(Output_ReconstructPath + 'reconstruct_image_' + str(img_no) +'.png', reconst) 
        
#--------------------------------------------------------
Input_StegoPath = "path to load stego images";
Output_ReconstructPath = "path to save reconstructed images";
image_Stego = []
image_Reconstruct = []
Stegoimg =0;
Reconstructimg =0;
#---------------------------------------------------------------------------------------------------
# Load Image
#---------------------------------------------------------------------------------------------------
valid_images = [".png"]
for fc in os.listdir(Input_StegoPath):
    ext = os.path.splitext(fc)[1]
    if ext.lower() not in valid_images:
        continue
    ims = imageio.v2.imread(os.path.join(Input_StegoPath,fc)).astype(float)/255
    ims = ims[:,:,0:3]
    image_Stego.append(ims)
    Stegoimg = Stegoimg +1;
  
#---------------------------------------------------------------------------------------------------
# Main Process
#--------------------------------------------------------------------------------------------------- 
for n in range(0,Stegoimg): 
    print('Process image no :',n)
    im_o = image_Stego[n]
    AjEd8bit_k_matrix_DecodeProcess(im_o,n)
    
