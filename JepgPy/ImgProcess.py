#!/usr/bin/env python

from glob import glob
import os

from PIL import Image

PHOTO_PATH = "F:/Photo/"
ORIGNAL_IMG_PATH = "Photos/original/"
CROPPED_IMG_PATH = "Photos/cropped/"


jpglist = glob(PHOTO_PATH + "**/*.[jJ][pP][gG]", recursive=True)
if not os.path.exists(CROPPED_IMG_PATH):
    os.makedirs(CROPPED_IMG_PATH)

if os.path.exists("Original_File_List.txt"):
    os.remove("Original_File_List.txt")
if os.path.exists("Cropped_File_List.txt"):
    os.remove("Cropped_File_List.txt")

#oflist = open("Original_File_List.txt", 'w')
oflist = []
cflist = open("Cropped_File_List.txt", 'w')

print("Total image count = " + str(len(jpglist)) + "\n")

if not len(jpglist):
    print("No input image\n")
    exit()

file_counter = 1
for jpg in jpglist:
    ofname = jpg.rsplit('\\', 1)[1].rsplit('.', 1)[0]   ## file_path\filename.JPG --> filename
    if(ofname in oflist):       ## Duplicate images should not be processed 
        continue

    try:
        im = Image.open(jpg)

        if (im.size == (4608, 3456)):                   ## Image 4:3 = 4608 * 3456 (GCD  = 1152)
            print("File " + str(file_counter) + " is proccessed")
            file_counter = file_counter + 1
            
            ## assume that the cropped size is 288 * 288
            X_Len = (int)(4608 / 288)   # = 16
            Y_Len = (int)(3456 / 288)   # = 12
            
            # oflist.write(ofname + '\n')
            im.save(ORIGNAL_IMG_PATH + ofname + '.JPG',  quality = 100)
            oflist.append(ofname)

    
            counter = 0
            for i in range(X_Len):
                for j in range(Y_Len):
                    nim = im.crop((i * 288, j * 288, (i + 1) * 288, (j + 1) * 288))
                    cfname = jpg.rsplit('.', 1)[0].rsplit('\\', 1)[1] + "_" + str(counter) + ".jpg"
                    nim.save( CROPPED_IMG_PATH + cfname, quality = 100)
                    cflist.write(cfname + '\n')
                    counter = counter + 1
    except:
        print(jpg + " is not identified")
                #nim.append(im.crop((j * 288, i * 288, (j + 1) * 288, (i + 1) * 288)))        # crop():  4-tuple defining the left, upper, right, and lower pixel coordinate.
    
    #elif (im.size == (4608, 2592)):              # Image 16:9 = 4608 * 2592 (GCD = 288), GCD(1152, 288) = 8
    #    print("It's a 16:9 image\n")

cflist.close()
with open("Original_File_List.txt", 'w') as of:
     of.write('\n'.join(oflist))