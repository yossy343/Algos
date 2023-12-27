# Import functions and libraries

import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy import r_
#from matplotlib.pyplot import imread
import scipy.fftpack
import imageio
import os, os.path
import csv
import cv2 

Stego_path = "path to load stego images";
Reconstruct_Hidden_path = "path to save reconstructed images";

image_Stego_tiff = []
image_Stego_png = []
image_Hidden = []
image_Hidden2 = []
image_Reconstruct_Hidden = []

Stegoimg_tiff =0;
Stegoimg_png =0;
hiddenimg = 0;
#---------------------------------------------------------------------------------------------------
# DCT function
#---------------------------------------------------------------------------------------------------
def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')
#---------------------------------------------------------------------------------------------------
# Load Image
#---------------------------------------------------------------------------------------------------
valid_images1 = [".tiff"]
 
for fs1 in os.listdir(Stego_path):
    ext = os.path.splitext(fs1)[1]
    if ext.lower() not in valid_images1:
        continue               
    image_Stego_tiff.append(imageio.v2.imread(os.path.join(Stego_path,fs1)).astype('float'))
    Stegoimg_tiff = Stegoimg_tiff +1;   
#---------------------------------------------------------------------------------------------------
# Main Process
#--------------------------------------------------------------------------------------------------- 
Reconstructed_FromTIFF = [];
Reconstructed_FromTIFF_Load = [];
Reconstructed_FromPNG = [];    

for n in range(0,Stegoimg_tiff): 
    imhtemp_tiff = image_Stego_tiff[n]
    imhtemp_png  = image_Stego_png[n]
    imh_tiff = imhtemp_tiff[:,:,0:3]
    imh_png = imhtemp_png[:,:,0:3]
       
    imsize = imh_tiff.shape
    dct_out_tiff = np.zeros(imsize)
    dct_out_png  = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            dct_out_tiff[i:(i+8),j:(j+8)] = dct2( imh_tiff[i:(i+8),j:(j+8)] )
            
    h8 = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            h8[i:(i+6),j:(j+6)] = dct_out_tiff[(i+2):(i+8),(j+2):(j+8)]

    reconst_tiff = np.zeros(imsize)
    alpha = 200
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            reconst_tiff[i:(i+8),j:(j+8)] = idct2( h8[i:(i+8),j:(j+8)]*alpha )
    reconst_tiff[reconst_tiff < 0] = 0
    reconst_tiff[reconst_tiff > 1] = 1
    
    Reconstructed_FromTIFF.append(reconst_tiff)
    
    if n < 10:
       imageio.imwrite(Reconstruct_Hidden_path + 'reconstruct_tiff_00' + str(n) +'.png', reconst_tiff*255)   
    elif n < 100 and n > 9:
       imageio.imwrite(Reconstruct_Hidden_path + 'reconstruct_tiff_0' + str(n) +'.png', reconst_tiff*255)   
    else:
       imageio.imwrite(Reconstruct_Hidden_path + 'reconstruct_tiff_' + str(n) +'.png', reconst_tiff*255)
   