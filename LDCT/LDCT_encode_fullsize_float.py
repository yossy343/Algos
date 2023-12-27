# This version keeps DCT of the first two rows and columns
# Hidden pixel values are downscale by a factor of alpha
# Write output with imageio.imwrite
import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy import r_
import os, os.path
import scipy.fftpack
import imageio
import cv2
import csv
Cover_path  = "path to load cover images";
Hidden_path = "path to load hidden images";
Output_StegoPath = "path to save stego images";

image_Cover = []
image_Hidden = []
coverimg =0;
hiddenimg =0;

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
valid_images = [".png"]
for fc in os.listdir(Cover_path):
    ext = os.path.splitext(fc)[1]
    if ext.lower() not in valid_images:
        continue
    im1 = imageio.v2.imread(os.path.join(Cover_path,fc))
    im = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
    image_Cover.append(im.astype(float)/255)
    coverimg = coverimg +1;

for fh in os.listdir(Hidden_path):
    ext = os.path.splitext(fh)[1]
    if ext.lower() not in valid_images:
        continue
    image_Hidden.append(imageio.v2.imread(os.path.join(Hidden_path,fh)).astype(float)/255)
    hiddenimg = hiddenimg +1;
#---------------------------------------------------------------------------------------------------
# Main Process
#--------------------------------------------------------------------------------------------------- 
for n in range(0,coverimg): 
    print("Process image no. ", n)
    im = image_Cover[n]
    h_im = image_Hidden[n]
    hiddenim = h_im[:,:,0:3]
    f = plt.figure()
    plt.imshow(im,cmap='gray')

    imsize = im.shape
    print(imsize);
    print(h_im.shape[0]);
    print(h_im.shape[1]);
    print(h_im.shape[2]);
    print("Covert RGB to DCT of Cover image no. ", n)
    dct = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            dct[i:(i+8),j:(j+8)] = dct2( im[i:(i+8),j:(j+8)] )

    newhidden = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            newhidden[i:(i+8),j:(j+8)] = dct2(hiddenim[i:(i+8),j:(j+8)])

    dct_stego = dct
    alpha = 200
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            dct_stego[(i+2):(i+8),(j+2):(j+8)] = newhidden[i:(i+6),j:(j+6)]/alpha
                 
    im_dct = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            im_dct[i:(i+8),j:(j+8)] = idct2( dct_stego[i:(i+8),j:(j+8)] )
    
    if n < 10:
        imageio.imwrite(Output_StegoPath + 'stego_image_00' + str(n) +'.tiff', im_dct)
        imageio.imwrite(Output_StegoPath + 'stego_image_00' + str(n) +'.png', im_dct)
    elif n < 100 and n > 9:
        imageio.imwrite(Output_StegoPath + 'stego_image_0' + str(n) +'.tiff', im_dct)
        imageio.imwrite(Output_StegoPath + 'stego_image_0' + str(n) +'.png', im_dct)
    else:
       imageio.imwrite(Output_StegoPath + 'stego_image_' + str(n) +'.tiff', im_dct)
       imageio.imwrite(Output_StegoPath + 'stego_image_' + str(n) +'.png', im_dct)
        