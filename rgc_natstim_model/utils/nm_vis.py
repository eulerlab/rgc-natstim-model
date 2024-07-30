import os
from skimage.measure import block_reduce
from skimage.transform import resize
import numpy as np

# to better visulaize image, use gamma correction to transfer image real to image view
def img_real2view(img):
    gamma_correction=lambda x:np.power(x,1.0/2.2)
    img_shape=img.shape
    # gray image
    if np.size(img_shape)==2:
        # uint8
        if np.max(img)>1:
            temp_view=np.zeros_like(img,dtype=np.float32)
            temp_view=np.float32(img)/255.0 # float32, 1.0
            temp_view=gamma_correction(temp_view)
            temp_view2=np.zeros_like(img,dtype=np.uint8)
            temp_view2=np.uint8(temp_view*255)
            return temp_view2
        # float
        if np.max(img)<2:
            return gamma_correction(img)
            
    # color image
    if np.size(img_shape)==3:
        # uint8, BGR
        if np.max(img)>1:
            temp_view=np.zeros_like(img,dtype=np.float32)
            temp_view=np.float32(img[...,::-1])/255.0 # gb,1.0
            temp_view[...,-1]=gamma_correction(temp_view[...,-1])
            temp_view[...,1]=gamma_correction(temp_view[...,1])
            temp_view2=np.zeros_like(img,dtype=np.uint8)
            temp_view2=np.uint8(temp_view[...,::-1]*255) # bgr,255
            return temp_view2
        # float, RGB
        if np.max(img)<2:
            return gamma_correction(img)

def downSample(img):
    # used in the Gui atm
    # input img is a 2D_matrix
    # 72x64->18x16

    output = np.zeros((18,16))
    output = block_reduce(img,block_size=(4,4),func = np.mean)

    return output
    
def rotate90(sti):
    '''
    rotate the stimuli clockwise for 90 degrees
    
    input:
        sti np.array: (4000,2,120,18,16)
    return
        r_sti np.array: (4000,2,120,16,18)
    '''
    sti= sti.reshape((4000*2*120,18,16))
    r_sti = np.rot90(sti, k=3,axes=(-1,-2))
    r_sti = r_sti.reshape((4000,2,120,16,18))
    return r_sti

def resize_downSample(img):
    # input img is a 2D_matrix3
    # 72x64->18x16

    output = resize(img,(18,16))
    return output
    
def get_rectangle_corners(xy_tuple):
    # Half the dimensions of the rectangle
    half_width = 64 / 2
    half_height = 72 / 2

    center_x,center_y = xy_tuple
    # Calculating the corner coordinates
    top_left = (int(center_x - half_width),int( center_y - half_height))
    top_right = (int(center_x + half_width), int(center_y - half_height))
    bottom_left = (int(center_x - half_width), int(center_y + half_height))
    bottom_right = (int(center_x + half_width), int(center_y + half_height))
    return top_left, top_right, bottom_left, bottom_right