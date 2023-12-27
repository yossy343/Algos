# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 10:29:01 2023

@author: COC
"""

# =============================================================================
# implementation of : Propose Methode HDCT
# =============================================================================

# Implement Code for Reference [22]
import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy import r_
import scipy.fftpack
import imageio
import cv2
import math
import os, os.path
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
from copy import deepcopy

# im = imageio.imread("lenna.png")
# im = im[:,:,0:3]
# h_im = imageio.imread("pepper_fullsize.png")
# h_im = h_im[:,:,0:3]
# f = plt.figure()
# plt.imshow(im,cmap='gray')
# plt.title("Cover image" )
# im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# him_gray = cv2.cvtColor(h_im, cv2.COLOR_BGR2GRAY)

Q8 = np.array([[16,11,10,16,24,40,51,61],      # QUANTIZATION TABLE
                    [12,12,14,19,26,58,60,55],    # required for DCT
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,103,99]])


# For Lumnance channel
Q8_1 = np.array([[16,11,10,16,24,40,51,61],      # QUANTIZATION TABLE
                    [12,12,14,19,26,58,60,55],    # required for DCT
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,103,99]])

Q8_3D = np.repeat(Q8_2[:, :, np.newaxis], 3, axis=2)

P_3D = np.repeat(P[:, :, np.newaxis], 3, axis=2)

Scale_Matrix = np.array([[1,  5,  5,   10,  20,  40,  80,  160],
					     [5,  5,  10,  20,  40,  80,  160, 320],
 						 [5,  10, 20,  40,  80,  160, 320, 1000],
				    	 [10, 20, 40,  80,  160, 320, 1000,1000],
 						 [20, 40, 80,  160, 320, 1000,1000,1000],
 						 [40, 80, 160, 320, 1000,1000,1000,1000],
 						 [80, 160,320, 1000,1000,1000,1000,1000],
 						 [160,320,1000,1000,1000,1000,1000,1000]])

Scale_Matrix_3D = np.repeat(Scale_Matrix[:, :, np.newaxis], 3, axis=2)
#------------------------------------------------------------------------------------------
Color_offset = 128
#------------------------------------------------------------------------------------------
def dct2(a):
    #return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0 , norm='ortho'), axis=1, norm='ortho')
#------------------------------------------------------------------------------------------
def idct2(a):
    #return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')
    return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1, norm='ortho')
#------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# Setup path of input and output
#--------------------------------------------------------------------------------------------------
Cover_path  = "path to load cover images";
Hidden_path = "path to load hidden images";
Output_StegoPath = "path to save stegano images";
Output_ReconstructPath = "path to save reconstruct hidden images";

image_Cover = []
image_Cover_Gray = []
image_Hidden = []
image_Stego = []

coverimg =0;
hiddenimg =0;
print("load cover image =  ",Cover_path);
print("load hidden image =  ",Hidden_path);
print("save stego image =  ",Output_StegoPath);
#---------------------------------------------------------------------------------------------------
# Load Image
#---------------------------------------------------------------------------------------------------
valid_images = [".png"]
w = 512
h = 512
dim = (w,h)

for fc in os.listdir(Cover_path):
    ext = os.path.splitext(fc)[1]
    if ext.lower() not in valid_images:
        continue
    imC = cv2.imread(os.path.join(Cover_path,fc))
    info = imC.shape
    if len(imC.shape) == 2:
        im_RGB = cv2.cvtColor(imC, cv2.COLOR_GRAY2BGR)
    else:
        im_RGB = imC
    im_RGB = cv2.resize(im_RGB, dim, interpolation = cv2.INTER_AREA)
    
    image_Cover.append(im_RGB)
    image_Cover_Gray.append(imC)
    coverimg = coverimg +1;

for fh in os.listdir(Hidden_path):
    ext = os.path.splitext(fh)[1]
    if ext.lower() not in valid_images:
        continue
    h_im = cv2.imread(os.path.join(Hidden_path,fh))
    h_im = cv2.resize(h_im, dim, interpolation = cv2.INTER_AREA)
    image_Hidden.append(h_im)
    hiddenimg = hiddenimg + 1;
   
print("Total cover image =  ",coverimg);
print("Total hidden image =  ",hiddenimg);
max_no = 0
if coverimg > hiddenimg:
     max_no = hiddenimg
else :
     max_no = coverimg
# =============================================================================
# Perform a blockwise DCT
for n in range(100,max_no):
    print("----------------------------------")
    print('Process Encode image no :',n)
    print("----------------------------------")
    im = image_Cover[n]
    h_im = image_Hidden[n]
    hiddenim = h_im[:,:,0:3]
    imsize = im.shape
    himsize = h_im.shape

    hidden_YCrCb = np.zeros(imsize)
    hidden_DCT = np.zeros(imsize)
    hidden_DCT_Scale = np.zeros(imsize)
    hidden_Dct_phase = np.zeros(imsize)
    quantizedDCT_Mag = np.zeros(imsize)
    quantizedDCT_Mag_Scale = np.zeros(imsize)
    quantizedDCT_Mag_ScaleInt = np.zeros(imsize)
    
    Sum_R1 = 0
    Sum_R2 = 0
    #-------------------------------------------------------------------------------------------
    # Do 8x8 DCT on hidden image (in-place)
    #-------------------------------------------------------------------------------------------
    offset_val = 0
    hiddenim = hiddenim.astype(float)
    hiddenim2 = hiddenim - offset_val
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            hidden_DCT[i:(i+8),j:(j+8)] = dct2( hiddenim2[i:(i+8),j:(j+8)])           
            quantizedDCT_Mag[i:(i+8),j:(j+8)] = abs(hidden_DCT[i:(i+8),j:(j+8)]/Q8_3D)  # Qualtization
            quantizedDCT_Mag_Scale[i:(i+8),j:(j+8)] = quantizedDCT_Mag[i:(i+8),j:(j+8)]*Scale_Matrix_3D  # Scale up
            np.round(quantizedDCT_Mag_Scale[i:(i+8),j:(j+8)],2)
    quantisedDCT_Mag_Scale = quantizedDCT_Mag_Scale 
    quantisedDCT_Mag_ScaleInt = deepcopy(np.round(quantizedDCT_Mag_Scale))
    hidden_Dct_phase = deepcopy(hidden_DCT)
    hidden_Dct_phase[hidden_Dct_phase >= 0] =  1
    hidden_Dct_phase[hidden_Dct_phase < 0]  = -1
    #-------------------------------------------------------------------------------------------     
    # Merge hidden DCT to Cover RGB
    #-------------------------------------------------------------------------------------------
    Output_Encode = np.zeros(imsize)
    Check_DCT = np.zeros(imsize)
    Check_DCT2 = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:       # row
        for j in r_[:imsize[1]:8]:   # col
             #  Copy hidden DCT from 4x4 top left 
             offset = 4         
             List_DCT_R = []
             List_DCT_G = []
             List_DCT_B = []
             SignDCT_R = []
             SignDCT_G = []
             SignDCT_B = []
             
             Check_DCT_R = 0
             Check_DCT_G = 0
             Check_DCT_B = 0
             #------------------------------------------------------------
             # Quardrand 1 of Hidden DCT n each block
             #------------------------------------------------------------
             Px = 16 
             Py = 48
             count = 0
             for sub_y in range(i + 0, i + 4, 1):
                 for sub_x in range(j + 0, j + 4 ,1):
                         DCT_val_R = int(round(quantisedDCT_Mag_ScaleInt[sub_y,sub_x][0]))
                         DCT_val_G = int(round(quantisedDCT_Mag_ScaleInt[sub_y,sub_x][1]))
                         DCT_val_B = int(round(quantisedDCT_Mag_ScaleInt[sub_y,sub_x][2]))
                         
                         Check_DCT2[sub_y,sub_x][0] = quantisedDCT_Mag_ScaleInt[sub_y,sub_x][0]
                         Check_DCT2[sub_y,sub_x][1] = quantisedDCT_Mag_ScaleInt[sub_y,sub_x][1]
                         Check_DCT2[sub_y,sub_x][2] = quantisedDCT_Mag_ScaleInt[sub_y,sub_x][2]
  
                         List_DCT_R.append(DCT_val_R)
                         List_DCT_G.append(DCT_val_G)
                         List_DCT_B.append(DCT_val_B)
                         
                         SignDCT_R.append(hidden_Dct_phase[sub_y,sub_x][0])
                         SignDCT_G.append(hidden_Dct_phase[sub_y,sub_x][1])
                         SignDCT_B.append(hidden_Dct_phase[sub_y,sub_x][2])                        
                        
                         if i == Py and j == Px:
                              print(quantisedDCT_Mag_ScaleInt[sub_y,sub_x][0]*hidden_Dct_phase[sub_y,sub_x][0], end="")
                              print("\t", end="")
                         count+=1
                 if i == Py and j == Px: 
                     print("\n")
             #------------------------------------------------------------------
             # Quardrand 1 of Cover
             #------------------------------------------------------------------
             count = 0
             for sub_y1 in range(i + 0, i + 4, 1):
                 for sub_x1 in range(j + 0, j + 4 ,1):
                         Cover_val_R = int(im[sub_y1,sub_x1][0])
                         Cover_val_G = int(im[sub_y1,sub_x1][1])
                         Cover_val_B = int(im[sub_y1,sub_x1][2])
                         
                         Sign_val_R = 0
                         Sign_val_G = 0
                         Sign_val_B = 0
                         
                         if SignDCT_R[count] == 1:
                            Sign_val_R = 4
                         if SignDCT_R[count] == -1:
                            Sign_val_R = 0
                            
                         if SignDCT_G[count] == 1:
                            Sign_val_G = 4
                         if SignDCT_G[count] == -1:
                            Sign_val_G = 0
                            
                         if SignDCT_B[count] == 1:
                            Sign_val_B = 4
                         if SignDCT_B[count] == -1:
                            Sign_val_B = 0   
                         
                         DCT_val_R = List_DCT_R[count]   
                         DCT_val_G = List_DCT_G[count] 
                         DCT_val_B = List_DCT_B[count] 
                         New_R = (Cover_val_R & 248) + (3 & (DCT_val_R  >> 6)) +  Sign_val_R
                         New_G = (Cover_val_G & 248) + (3 & (DCT_val_G  >> 6)) +  Sign_val_G
                         New_B = (Cover_val_B & 248) + (3 & (DCT_val_B  >> 6)) +  Sign_val_B
                         
                         Output_Encode[sub_y1,sub_x1][0] = New_R
                         Output_Encode[sub_y1,sub_x1][1] = New_G
                         Output_Encode[sub_y1,sub_x1][2] = New_B
                         
                         Check_DCT_R += ((3 & DCT_val_R) << 0)
                         Check_DCT_G += ((3 & DCT_val_G) << 0)
                         Check_DCT_B += ((3 & DCT_val_B) << 0)
                         count+=1                         
             #------------------------------------------------------------------
             # Quardrand 2 of Cover
             #------------------------------------------------------------------
             count = 0
             for sub_y2 in range(i + 0, i + 4, 1):
                 for sub_x2 in range(j + 4, j + 8 ,1):
                         Cover_val_R = int(im[sub_y2,sub_x2][0])
                         Cover_val_G = int(im[sub_y2,sub_x2][1])
                         Cover_val_B = int(im[sub_y2,sub_x2][2])
                         DCT_val_R = List_DCT_R[count]   
                         DCT_val_G = List_DCT_G[count] 
                         DCT_val_B = List_DCT_B[count] 
                         
                         New_R = (Cover_val_R & 252) + (3 & (DCT_val_R  >> 4)) 
                         New_G = (Cover_val_G & 252) + (3 & (DCT_val_G  >> 4))
                         New_B = (Cover_val_B & 252) + (3 & (DCT_val_B  >> 4))
                         
                         Output_Encode[sub_y2,sub_x2][0] = New_R
                         Output_Encode[sub_y2,sub_x2][1] = New_G
                         Output_Encode[sub_y2,sub_x2][2] = New_B
                         
                         Check_DCT_R += ((3 & DCT_val_R) << 2)
                         Check_DCT_G += ((3 & DCT_val_G) << 2)
                         Check_DCT_B += ((3 & DCT_val_B) << 2)
                         count+=1                    
             #------------------------------------------------------------------
             # Quardrand 3 of Cover
             #------------------------------------------------------------------
             count = 0
             for sub_y3 in range(i + 4 , i + 8, 1):
                 for sub_x3 in range(j + 0, j + 4 ,1):
                         Cover_val_R = int(im[sub_y3,sub_x3][0])
                         Cover_val_G = int(im[sub_y3,sub_x3][1])
                         Cover_val_B = int(im[sub_y3,sub_x3][2])
                         DCT_val_R = List_DCT_R[count]   
                         DCT_val_G = List_DCT_G[count] 
                         DCT_val_B = List_DCT_B[count] 
                         New_R = (Cover_val_R & 252) + (3 & (DCT_val_R  >> 2)) 
                         New_G = (Cover_val_G & 252) + (3 & (DCT_val_G  >> 2))
                         New_B = (Cover_val_B & 252) + (3 & (DCT_val_B  >> 2))
                         Output_Encode[sub_y3,sub_x3][0] = New_R
                         Output_Encode[sub_y3,sub_x3][1] = New_G
                         Output_Encode[sub_y3,sub_x3][2] = New_B
                         
                         Check_DCT_R += ((3 & DCT_val_R) << 4)
                         Check_DCT_G += ((3 & DCT_val_G) << 4)
                         Check_DCT_B += ((3 & DCT_val_B) << 4)
                         count+=1         
             #------------------------------------------------------------------
             # Quardrand 4 of Cover
             #------------------------------------------------------------------
             count = 0
             for sub_y4 in range(i + 4, i + 8, 1):
                 for sub_x4 in range(j + 4, j + 8 ,1):
                         Cover_val_R = int(im[sub_y4,sub_x4][0])
                         Cover_val_G = int(im[sub_y4,sub_x4][1])
                         Cover_val_B = int(im[sub_y4,sub_x4][2])
                         DCT_val_R = List_DCT_R[count]   
                         DCT_val_G = List_DCT_G[count] 
                         DCT_val_B = List_DCT_B[count] 
                         
                         New_R = (Cover_val_R & 252) + (3 & (DCT_val_R  >> 0)) 
                         New_G = (Cover_val_G & 252) + (3 & (DCT_val_G  >> 0))
                         New_B = (Cover_val_B & 252) + (3 & (DCT_val_B  >> 0))

                         Output_Encode[sub_y4,sub_x4][0] = New_R
                         Output_Encode[sub_y4,sub_x4][1] = New_G
                         Output_Encode[sub_y4,sub_x4][2] = New_B
                         
                         Check_DCT_R += ((3 & DCT_val_R) << 6)
                         Check_DCT_G += ((3 & DCT_val_G) << 6)
                         Check_DCT_B += ((3 & DCT_val_B) << 6)
                         
                         count+=1

             count = 0
             
             count = 0
             for sub_y in range(i + 0, i + 4, 1):
                 for sub_x in range(j + 0, j + 4 ,1):
                     Check_DCT[sub_y,sub_x][0] = List_DCT_R[count]
                     Check_DCT[sub_y,sub_x][1] = List_DCT_G[count]
                     Check_DCT[sub_y,sub_x][2] = List_DCT_B[count]
                     count += 1

    if n < 10:
        cv2.imwrite(Output_StegoPath + 'stego_image_00'+ str(n) + '.png', Output_Encode)
    elif n < 100 and n > 9:
        cv2.imwrite(Output_StegoPath + 'stego_image_0'+ str(n) + '.png', Output_Encode)
    else:
        cv2.imwrite(Output_StegoPath + 'stego_image_'+ str(n) + '.png', Output_Encode)
    
    
    print("----------------------------------")
    print('Process Decode image no :',n)
    print("----------------------------------")
    
    stego_im = cv2.imread(Output_StegoPath + "stego_image_00" + str(n) +".png", cv2.IMREAD_COLOR)
    if n < 10:
        stego_im = cv2.imread(Output_StegoPath + "stego_image_00" + str(n) +".png", cv2.IMREAD_COLOR)
    elif n < 100 and n > 9:
        stego_im = cv2.imread(Output_StegoPath + "stego_image_0" + str(n) +".png", cv2.IMREAD_COLOR)
    else:
        stego_im = cv2.imread(Output_StegoPath + "stego_image_" + str(n) +".png", cv2.IMREAD_COLOR)
        
    imsize = stego_im.shape
    Output_Decode_DCT = np.zeros(imsize)
    SignMap = np.ones(imsize)
    
    for i in r_[:imsize[0]:8]:       # row
        for j in r_[:imsize[1]:8]:   # col
              for sub_y in range(i + 0, i + 4, 1):
                  for sub_x in range(j + 0, j + 4 ,1):
                      # Calculae Sign number
                      Restore_R = 0
                      Restore_G = 0
                      Restore_B = 0
                      
                      Val_R_Q1 = int(stego_im[sub_y + 0,sub_x + 0][0])
                      Val_G_Q1 = int(stego_im[sub_y + 0,sub_x + 0][1])
                      Val_B_Q1 = int(stego_im[sub_y + 0,sub_x + 0][2])
                      
                      if Val_R_Q1 & 4 == 4:
                          SignMap[sub_y,sub_x][0] = 1
                      if Val_R_Q1 & 4 == 0:
                          SignMap[sub_y,sub_x][0] = -1
                      
                      if Val_G_Q1 & 4 == 4:
                          SignMap[sub_y,sub_x][1] = 1
                      if Val_G_Q1 & 4 == 0:
                          SignMap[sub_y,sub_x][1] = -1
                      
                      if Val_B_Q1 & 4 == 4:
                          SignMap[sub_y,sub_x][2] = 1
                      if Val_B_Q1 & 4 == 0:
                          SignMap[sub_y,sub_x][2] = -1
                      
                      Val_R_Q2 = int(stego_im[sub_y + 0,sub_x + 4][0])
                      Val_G_Q2 = int(stego_im[sub_y + 0,sub_x + 4][1])
                      Val_B_Q2 = int(stego_im[sub_y + 0,sub_x + 4][2])
                      
                      Val_R_Q3 = int(stego_im[sub_y + 4,sub_x + 0][0])
                      Val_G_Q3 = int(stego_im[sub_y + 4,sub_x + 0][1])
                      Val_B_Q3 = int(stego_im[sub_y + 4,sub_x + 0][2])
                      
                      Val_R_Q4 = int(stego_im[sub_y + 4,sub_x + 4][0])
                      Val_G_Q4 = int(stego_im[sub_y + 4,sub_x + 4][1])
                      Val_B_Q4 = int(stego_im[sub_y + 4,sub_x + 4][2])
                      
                      Restore_R = ((Val_R_Q1 & 3) << 6) + ((Val_R_Q2 & 3) << 4) + ((Val_R_Q3 & 3) << 2) + ((Val_R_Q4 & 3) << 0)
                      Restore_G = ((Val_G_Q1 & 3) << 6) + ((Val_G_Q2 & 3) << 4) + ((Val_G_Q3 & 3) << 2) + ((Val_G_Q4 & 3) << 0) 
                      Restore_B = ((Val_B_Q1 & 3) << 6) + ((Val_B_Q2 & 3) << 4) + ((Val_B_Q3 & 3) << 2) + ((Val_B_Q4 & 3) << 0) 
                      
                      Output_Decode_DCT[sub_y,sub_x][0] = Restore_R
                      Output_Decode_DCT[sub_y,sub_x][1] = Restore_G
                      Output_Decode_DCT[sub_y,sub_x][2] = Restore_B 
                      
                      if i == Py and j == Px:
                           print(Output_Decode_DCT[sub_y,sub_x][0]*SignMap[sub_y,sub_x][0], end="")
                           print("\t", end="")
                      count+=1
                  if i == Py and j == Px: 
                      print("\n")  
                  
    Output_Decode_DCT = Output_Decode_DCT.astype(float)
    De_quantize = np.zeros(imsize)
    De_ScaleUp = np.zeros(imsize)
    
    Restore_Data = np.zeros(imsize)
    reconst = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:            
            reconst[i:(i+8),j:(j+8)] = idct2(((Output_Decode_DCT[i:(i+8),j:(j+8)]/Scale_Matrix_3D)*Q8_3D)*SignMap[i:(i+8),j:(j+8)])
    cv2.imwrite("Reconstruct_image_1.png", reconst)
    
    if n < 10:
        cv2.imwrite(Output_ReconstructPath + "Reconstruct_image_00" + str(n) +".png", reconst)
    elif n < 100 and n > 9:
        cv2.imwrite(Output_ReconstructPath + "Reconstruct_image_0" + str(n) +".png", reconst)
    else:
        cv2.imwrite(Output_ReconstructPath + "Reconstruct_image_" + str(n) +".png", reconst)
    