import numpy as np
import os
from PIL import Image

##############################################################################################
#
#   Image processing for loading or storing image and tranform between Array and Jpeg object
#
##############################################################################################


### Global Variables ###
CROPPED_IMG_PATH = "./JepgPy/Photos/cropped/"
ORIGINAL_IMG_PATH = "./JepgPy/Photos/original/"
ORIGINAL_FLIST_PATH = "./JepgPy/Original_File_List.txt"

## Slice a image into SWid * SHei pieces of image
global  SWid
global  SHei
global  FILE_LOAD
FILE_LOAD = 5  # number of cropped image loaded into memory at once (consider memory size limitation)
SHei = 18
SWid = 18

def GetOriginalFileList():
    if os.path.exists(ORIGINAL_FLIST_PATH):
        f = open(ORIGINAL_FLIST_PATH, 'r')
    else:
        return None
    flist = f.read().splitlines()
    f.close()
    return flist

def OriToCropFname(OriFname, number):
    return OriFname.rsplit('.', 1)[0] + '_' + str(number) + '.jpg'

def SliceOriImage(OriFname):
    try:
        im = Image.open(ORIGINAL_IMG_PATH + OriFname)
        Width, Height = im.size
        data = []
        #if (im.size == (4608, 3456)):
        X_Len = int(Width / SWid)
        Y_Len = int(Height / SHei)
        for j in range(Y_Len):
            for i in range(X_Len):
                nim = im.crop((i * SWid, j * SHei, (i + 1) * SWid, (j + 1) * SHei))
                arr = np.asarray(nim, dtype = 'float32')
                data.append(np.reshape(arr, (arr.shape[0] * arr.shape[1] * arr.shape[2])))
        im.close()
    except:
        data = None
    return data
                    
def ShowImage(filename):
    Img = Image.open(filename)
    Img.show()



def LoadCroppedImage(flist, OriIndexFrom, OriIndexTo, shuffle=False):
    # Since a original image is about 160M pixels and 12 bytes (float32 for R,G,B value) for each pixel, 192MB memory space is require for an image.
    # That is 1MB memory for a cropped image, we load 3000 cropped image each time. (total 3GB memory )
    
    if flist is None:
        print("\nError: Original_File_List.txt is not found.\n")
        exit(1)
    
    FLen = len(flist)
    
    if (OriIndexTo > FLen):       #Invalid parameter: index out of range
        print("\nInvalid parameter: Index out of range\n")
        return None
    if(OriIndexFrom > OriIndexTo):
        print("\nInvalid parameter\n")
        return None

    data = []
    for fname in flist[OriIndexFrom:OriIndexTo + 1]:
        Img = SliceOriImage(fname)
        if Img is not None:
            data = data + Img
           
    data = np.asarray(data, dtype = np.float32)

    if shuffle:
        np.random.shuffle(data)
    return data


def ShiftedCroppedImage(OriFname):
    try:
        im = Image.open(ORIGINAL_IMG_PATH + OriFname)
        im = im.crop((SWid / 2, SHei / 2, im.size[0] - SWid / 2, im.size[1] - SHei / 2))
        Width, Height = im.size

        data = []
        #if (im.size == (4608, 3456)):
        X_Len = int(Width / SWid)   # = 16
        Y_Len = int(Height / SHei)   # = 12)
        for j in range(Y_Len):
            for i in range(X_Len):
                nim = im.crop((i * SWid, j * SHei, (i + 1) * SWid, (j + 1) * SHei))
                arr = np.asarray(nim, dtype = 'float32')
                data.append(np.reshape(arr, (arr.shape[0] * arr.shape[1] * arr.shape[2])))

        data = np.asarray(data, dtype = np.float32)
    except:
        data = None
    return data