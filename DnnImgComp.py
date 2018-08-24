from Token import *
from ImgToArrayLib import *
from AutoEncoder import *
from ThreadClassDef import ThreadWithReturnValue
import tkinter              # PYTHON 3 ONLY
from PIL import Image

if TESTING_IMAGE_ENABLE:
    from TestImageLib import *

if ASK_FOR_SAVE_MODEL_ENABLE:
    from QueryLib import query_yes_no


# Config the matplotlib backend as plotting inline in IPython
#%matplotlib inline

##################################################
#
#   Machine declaration
#
##################################################

## Since a piece of sliced image is 12 * 12 pixels, the input data matrice should be BATCH_SIZE * N_Shape
if __name__ == "__main__":

    print("Build AutoEncoder")
    N_Shape = SWid * SHei * 3
    model_1 = Autoencoder( n_features=N_Shape,
                        #learning_rate= 0.0005,    ## last best value is 0.0005 (loss mean 2000 min 1600); 0.005 all black
                        learning_rate= 0.0005,
                        n_hidden=[N_Shape, 500, 250, 125],
                        #n_hidden=[N_Shape, 500, 200],
                        #alpha=0.00,
                        alpha=0.000,   # ASUS-Joseph-18081303
                        decay_rate = 0.99      # 1 means no decay
                        )



    ##################################################
    #
    #   Training process
    #
    ##################################################


    ### Global Variables ###

    flist = GetOriginalFileList()

    if TRANING_MODE_ENABLE:

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
                    validation_data=valid_data,
                    #test_data=(test_data,test_data),
                    test_data = None,
                    batch_size = BATCH_SIZE,
                    )

            train_data = Cim.join()

            if TESTING_IMAGE_ENABLE:
                if i % (NumImgTrainedB4Test / FILE_LOAD) == 0:
                    print("\nPredict images, %d images have been trained ..." %(i * FILE_LOAD))
                    PredictTestFileAndShow(model_1 = model_1, flist = flist, findex = 0, file_no = i * FILE_LOAD)

            if AUTO_SAVE_MODEL_ENABLE:
                Respond = True
                if ASK_FOR_SAVE_MODEL_ENABLE:
                    Respond = query_yes_no("Save current model ?")

                if Respond and (i % (NumImgTrainedB4Test / FILE_LOAD) == 0):
                    model_1.save_model(EXPORT_PATH)
        
        if FINISH_SAVE_MODEL_ENABLE:
            model_1.save_model(EXPORT_PATH)


    """
    fig, axis = plt.subplots(2, 15, figsize=(15, 2))
    for i in range(0,15):

        img_original = np.reshape(test_data.images[i],(28,28))
        axis[0][i].imshow(img_original, cmap='gray')
        img = np.reshape(model_1.predict(test_data.images[i]),(28,28))
        axis[1][i].imshow(img, cmap='gray')
    plt.show()
    """
    

    if TESTING_IMAGE_ENABLE:
        PredictTestFileAndShow(model_1 = model_1, flist = flist, findex = 0)

    ##########################
    #
    #   Complete Message Box
    #
    ##########################

    
    
    root = tkinter.Tk()
    root.wm_attributes('-topmost',1)
    root.title('Program Completed')
    tkinter.Label(root, text = "The testing image: is created\nPlease close this windows.", font=("Courier", 20)).grid()

    root.mainloop()

    #fig, axis = plt.subplots(1, 2, figsize=(2, 1))
