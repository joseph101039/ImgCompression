import numpy as np
import os
from PIL import Image



CROPPED_IMG_PATH = "./JepgPy/Photos/cropped/"
ORIGINAL_IMG_PATH = "./JepgPy/Photos/original/"
ORIGINAL_FLIST_PATH = "./JepgPy/Original_File_List.txt"
NUM_CROP_PER_IMAGE = 192

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
    im = Image.open(ORIGINAL_IMG_PATH + OriFname)
    data = []
    if (im.size == (4608, 3456)):
        X_Len = (int)(4608 / 288)   # = 16
        Y_Len = (int)(3456 / 288)   # = 12)
        for j in range(Y_Len):
            for i in range(X_Len):
                nim = im.crop((i * 288, j * 288, (i + 1) * 288, (j + 1) * 288))
                arr = np.asarray(nim, dtype = 'float32')
                data.append(np.reshape(arr, (arr.shape[0], arr.shape[1]*arr.shape[2])))
    return data
                    



def LoadCroppedImage(flist, OriIndexFrom, OriIndexTo, shuffle=True):
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
        data = data + SliceOriImage(fname)
           
    data = np.asarray(data, dtype = np.float32)
    if shuffle:
        np.random.shuffle(data)
    return data

def ShowImage(filename):
    Img = Image.open(filename)
    Img.show()