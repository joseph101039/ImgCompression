from ImgArray import *
from AutoEncoder import *
from threading import Thread    # ASUS-Joseph-18081101

from tkinter import *               # PYTHON 3 ONLY

from PIL import Image



# ASUS-Joseph-18081101 >>>
class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self):
        Thread.join(self)
        return self._return

############################
#
#       Prediction
#
############################
### Global variables ###
## Slice a image into SWid * SHei pieces of image
N_Shape = SWid * SHei * 3

def PredictTestFileAndShow(flist, findex = 0):

    ############# notice DPI and save file may change content (JPEG compression)#################
    X_Len = (int)(4608 / SWid)   # = 16
    Y_Len = (int)(3456 / SHei)   # = 12

    TestFileIndex = 0

    AssImg = Image.new('RGB', (4608, 3456), (255, 255, 255))
    test_data = LoadCroppedImage(flist, TestFileIndex,TestFileIndex, False) # shape = (384, 288, 864)

    PredData = model_1.predict(test_data)
	### Fix the RGB value overflow error when ReLU transform function is selected for output layer.
	### ReLU does not set the upper bound value.
    PredData = PredData.clip(max=255, min=0)


    for i in range(X_Len * Y_Len):
        Pred = np.reshape(PredData[i], (SWid, SHei, 3))
        #print(Pred[0])         ####
        #os.system("pause")  ####
        Cim = Image.fromarray(np.asarray(Pred, dtype=np.int8), mode="RGB")
        Left = (i % X_Len) * SWid
        Upper = int(i / X_Len) * SHei
        AssImg.paste(Cim, (Left, Upper, Left + SWid, Upper + SHei))      ## a 4-tuple defining the left, upper, right, and lower pixel coordinate
        
    PredFile = flist[TestFileIndex].rsplit('.', 1)[0] + "_pred_" +  str(findex) + ".jpg"
    AssImg.save(PredFile, quality = 100)
    ShowImage(PredFile)


# ASUS-Joseph-18081101 <<<


# Config the matplotlib backend as plotting inline in IPython
#%matplotlib inline


##################################################
#
#   Machine declaration
#
##################################################

## Since a piece of sliced image is 12 * 12 pixels, the input data matrice should be BATCH_SIZE * N_Shape

print("Build AutoEncoder")

model_1 = Autoencoder( n_features=N_Shape,
                     #learning_rate= 0.0005,    ## last best value is 0.0005 (loss mean 2000 min 1600); 0.005 all black
                     learning_rate= 0.0005,
                     n_hidden=[N_Shape, 400, 300],
                     #alpha=0.00,
                     alpha=0.001,   # ASUS-Joseph-18081303
                     decay_rate = 0.99      # 1 means no decay
                    )



##################################################
#
#   Training process
#
##################################################


### Global Variables ###
FILE_LOAD = 5  # number of cropped image loaded into memory at once (consider memory size limitation)

flist = GetOriginalFileList()
valid_data = LoadCroppedImage(flist, 0, 0, False)
print("Start training")


Cim = ThreadWithReturnValue(target=LoadCroppedImage, args=(flist, 0, FILE_LOAD - 1, True))
Cim.start()
train_data = Cim.join()



for i in range(1,int(len(flist) / FILE_LOAD)):
    Cim = ThreadWithReturnValue(target=LoadCroppedImage, args=(flist, i * FILE_LOAD, (i+1) * FILE_LOAD - 1, True))        # ASUS-Joseph-18081101
    Cim.start()

    print("Training data: Image %d" %(i * FILE_LOAD))
    model_1.fit(X=train_data,
            Y=train_data,
            epochs=1,  
            #epochs=40,   ## ASUS-Joseph-18080601
            validation_data=valid_data,
            #test_data=(test_data,test_data),
            test_data = None,
            batch_size = 500,
            )

    train_data = Cim.join()

    if (i % 10) == 0:
        PredictTestFileAndShow(flist, i * FILE_LOAD)

"""
fig, axis = plt.subplots(2, 15, figsize=(15, 2))
for i in range(0,15):

    img_original = np.reshape(test_data.images[i],(28,28))
    axis[0][i].imshow(img_original, cmap='gray')
    img = np.reshape(model_1.predict(test_data.images[i]),(28,28))
    axis[1][i].imshow(img, cmap='gray')
plt.show()
"""



##########################
#
#   Complete Message Box
#
##########################

PredictTestFileAndShow(flist, 0)

#SI = ThreadWithReturnValue(target=ShowImage,args=('FILE0036_pred.jpg',))
#SI.start()
#SI.join()

root = Tk()
root.wm_attributes('-topmost',1)
root.title('Program Completed')
Label(root, text = "The testing image: " + PredFile + " is created\nPlease close this windows.", font=("Courier", 20)).grid()

root.mainloop()

#fig, axis = plt.subplots(1, 2, figsize=(2, 1))
