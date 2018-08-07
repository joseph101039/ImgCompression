import numpy as np
import os
from PIL import Image

CROPPED_IMG_PATH = "Photos/cropped/"

def GetCroppedFileList():
    if os.path.exists("Cropped_File_List.txt"):
        f = open("Cropped_File_List.txt", 'w')
    else:
        return None
    flist = f.read().splitlines()
    return flist

def LoadCroppedImage():
    # Since a original image is about 160M pixels and 12 bytes (float32 for R,G,B value) for each pixel, 192MB memory space is require for an image.
    # That is 1MB memory for a cropped image, we load 3000 cropped image each time. (total 3GB memory )
    flist = GetCroppedFileList()
    if flist is None:
        exit(1)
    
    FLen = len(flist)
    FIndex = 3000
    FEnd = False

    if FIndex > FLen:
        FIndex = FLen
        FEnd = True

    if not FEnd:
        for fname in flist[FIndex - 3000:FIndex]:
            im = Image.open(CROPPED_IMG_PATH + fname)
            arr = np.asarray(im, dtype = float32)
            arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2]) ## shape = (288, 288, 3)

            ### Image.getdata(), Image.putdata ?

    else:



    