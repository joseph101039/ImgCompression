import numpy as np
import os
from PIL import Image



CROPPED_IMG_PATH = "./JepgPy/Photos/cropped/"
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


def LoadCroppedImage(flist, OriIndexFrom, OriIndexTo):
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
        for num in range(NUM_CROP_PER_IMAGE):
            im = Image.open(CROPPED_IMG_PATH + OriToCropFname(fname, num))
            arr = np.asarray(im, dtype = np.float32)
            arr = arr.reshape((arr.shape[0], arr.shape[1] * arr.shape[2])) ## shape = (288, 288, 3)
            data.append(arr)
            ### Image.getdata(), Image.putdata ?
        
    data = np.asarray(data, dtype = np.float32)
    return data

    