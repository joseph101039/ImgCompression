#!/usr/bin/env python

from glob import glob
import os
from PIL import Image
import time, threading, datetime

PHOTO_PATH = "F:/Photo/"
ORIGINAL_IMG_PATH = "Photos/original/"
CROPPED_IMG_PATH = "Photos/cropped/"


if not os.path.exists(CROPPED_IMG_PATH):
    os.makedirs(CROPPED_IMG_PATH)

if os.path.exists("Original_File_List.txt"):
    os.remove("Original_File_List.txt")
if os.path.exists("Cropped_File_List.txt"):
    os.remove("Cropped_File_List.txt")

oflist = []




def SliceImage(jpglist, name):
    file_counter = 1
    for jpg in jpglist:
        ofname = jpg.rsplit('\\', 1)[1].rsplit('.', 1)[0]   ## file_path\filename.JPG --> filename
        if(ofname in oflist):       ## Duplicate images should not be processed 
            continue

        try:
            im = Image.open(jpg)

            if (im.size == (4608, 3456)):                   ## Image 4:3 = 4608 * 3456 (GCD  = 1152)
                print(name + ": File " + str(file_counter) + " is proccessed")
                file_counter = file_counter + 1
                
                ## assume that the cropped size is 288 * 288
                X_Len = (int)(4608 / 288)   # = 16
                Y_Len = (int)(3456 / 288)   # = 12
                
                # oflist.write(ofname + '\n')
                im.save(ORIGINAL_IMG_PATH + ofname + '.JPG',  quality = 100)
                oflist.append(ofname+ '.JPG')

        
                counter = 0
                for j in range(Y_Len):
                    for i in range(X_Len):
                        nim = im.crop((i * 288, j * 288, (i + 1) * 288, (j + 1) * 288))
                        cfname = jpg.rsplit('.', 1)[0].rsplit('\\', 1)[1] + "_" + str(counter) + ".jpg"
                        nim.save( CROPPED_IMG_PATH + cfname, quality = 100)
                        #cflist.write(cfname + '\n')
                        counter = counter + 1
        except:
            print(jpg + " is not identified")
                    #nim.append(im.crop((j * 288, i * 288, (j + 1) * 288, (i + 1) * 288)))        # crop():  4-tuple defining the left, upper, right, and lower pixel coordinate.
        
        #elif (im.size == (4608, 2592)):              # Image 16:9 = 4608 * 2592 (GCD = 288), GCD(1152, 288) = 8
        #    print("It's a 16:9 image\n")


if __name__ == '__main__':

	# Start activity to digest queue.  
	st = datetime.datetime.now()  

	TotalJpglist = glob(PHOTO_PATH + "**/*.[jJ][pP][gG]", recursive=True)
	JLen = len(TotalJpglist)
	print("Total image count = " + str(JLen) + "\n")

	if not JLen:
	    print("No input image\n")
	    exit()


	## Open three threads  
	thd1 = threading.Thread(target=SliceImage, name='Thd1', args=(TotalJpglist[:(int)(JLen / 3)], "T1"))  
	thd2 = threading.Thread(target=SliceImage, name='Thd2', args=(TotalJpglist[int(JLen / 3) + 1 : int(JLen*2 / 3)],"T2"))  
	thd3 = threading.Thread(target=SliceImage, name='Thd3', args=(TotalJpglist[int(JLen*2 / 3):], "T3")) 


	thd1.start()  
	thd2.start()  
	thd3.start()

	## Wait for all threads to terminate.  
	thd1.join()
    thd2.join()
    thd3.join()
	
	td = datetime.datetime.now() - st  
	print("\t[Info] Spending time={0}!".format(td)) 


	with open("Original_File_List.txt", 'w') as of:
	     of.write('\n'.join(oflist))