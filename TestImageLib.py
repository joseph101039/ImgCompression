from Token import *
from ImgToArrayLib import *
import numpy as np
from ThreadClassDef import ThreadWithReturnValue
if DENOISE_USING_THREADS:
    import time


def DenoiseImgArray(ImgSeg): # argument ImgSeg(SHei * SWid * 3)
    RgbMean = np.mean(np.mean(ImgSeg, axis=1), axis=0)
    ArrMean = np.asarray([[RgbMean for i in range(SWid)] for j in range(SHei)])
    SqArr = np.sqrt(np.sum(np.square(np.subtract(ImgSeg, ArrMean)), axis = 2))    # Calculate the square sum and take square root
    TwoStd = 2 * np.std(SqArr)
    
    ### When a pixels is out of 2 standard deviation, take it as a noise dot
    DenoiseArr = np.copy(ImgSeg)
    NotNoise = np.asarray([[[(SqArr[i, j] < TwoStd).astype(int) for k in range(3)] for j in range(SWid)] for i in range(SHei)]) # If out of 2 stdev, value = 0; otherwise 1

    for i in range(SHei):
        for j in range(SWid):
            startX, endX = j - 1, j + 2
            startY, endY = i - 1, i + 2
            if not NotNoise[i, j, 0]:
                if j == 0:              startX = 0
                elif j == (SWid - 1):   endX = SWid
                if i == 0:              startY = 0
                elif i == (SHei - 1):   endY = SHei
                SamplesNum = (np.sum(NotNoise[startY:endY, startX:endX]) / 3)
                #print("(%d, %d) -> (%d, %d): %d samples" %(startY, startX, endY, endX, int(SamplesNum)))
                if SamplesNum:  # avoid divide-zero error
                    Mul = np.multiply(ImgSeg[startY:endY, startX:endX], NotNoise[startY:endY, startX:endX])
                    DenoiseArr[i, j] = np.sum(np.sum(Mul, axis=1),axis=0)  / SamplesNum

    #DenoiseArr = np.asarray(np.around(DenoiseArr))    # floating number around follows IEEE754
    return  np.asarray(DenoiseArr)

if DENOISE_USING_THREADS:
    def ThreadingDenoise(PredRowData, i):
        start = time.clock()        # ASUS-Joseph-test
        dataX = []
        for data in PredRowData:
            Seg = np.reshape(data, (SHei, SWid, 3))
            Seg = DenoiseImgArray(Seg)
            dataX.append(Seg)
        
        dataX = np.concatenate((dataX), axis=1)
        print("thread %d time: " %(i))      # ASUS-Joseph-test
        print(time.clock() - start)     # ASUS-Joseph-test
        return dataX



# ASUS-Joseph-18082202 >>>
def BlendShiftedPredData(model_1, PredData, OriFname):
    ShiftImg = ShiftedCroppedImage(OriFname)
    PredShiData = (model_1.predict(ShiftImg)).clip(max=255, min=0)

    X_Len = int(4608 / SWid) - 1
    Y_Len = int(3456 / SHei) - 1

    dataY = []
    if DENOISE_USING_THREADS:
        ### Threading method
        for j in range(int(Y_Len / DenoiseTNum)): ### create 16 threads
            thread = []
            for k in range(DenoiseTNum):
                thread.append(ThreadWithReturnValue(target=ThreadingDenoise, args=(PredShiData[(j*DenoiseTNum+k) * X_Len:((j+1)*DenoiseTNum+k)*X_Len], k,)))
                thread[k].start()

            for k in range(DenoiseTNum):
                dataY.append(thread[k].join())
    else:
        ### serial method
        for j in range(Y_Len):
            dataX = []
            for i in range(X_Len):
                Seg = np.reshape(PredShiData[i + j * X_Len], (SHei, SWid, 3))
                Seg = DenoiseImgArray(Seg)
                dataX.append(Seg)
            
            #dataX = (np.asarray(list(zip(*dataX)))).reshape(SHei, 4608, 3)
            dataX = np.concatenate((dataX), axis=1)
            dataY.append(dataX)
    
    PredShiData = np.asarray(dataY)
    PredShiData = np.reshape(PredShiData, (3456 - SHei, 4608 - SWid, 3))
    PredShiData += PredData[ int(SHei / 2):int(3456 - SHei / 2), int(SWid / 2):int(4608 - SWid / 2)] 
    PredShiData = np.divide(PredShiData, 2)
    PredData[int(SHei / 2):int(3456 - SHei / 2), int(SWid / 2):int(4608 - SWid / 2 )] = PredShiData
    return PredData
# ASUS-Joseph-18082202 <<<


def PredictTestFileAndShow(model_1, flist, findex = 0):

    ############# notice DPI and save file may change content (JPEG compression)#################
    X_Len = (int)(4608 / SWid)   # = 16
    Y_Len = (int)(3456 / SHei)   # = 12

    TestFileIndex = 0

    #AssImg = Image.new('RGB', (4608, 3456), (255, 255, 255))
    test_data = LoadCroppedImage(flist, TestFileIndex,TestFileIndex, False) # shape = (384, 288, 864)

    PredData = model_1.predict(test_data)
    ### Fix the RGB value overflow error when ReLU transform function is selected for output layer.
    ### ReLU does not set the upper bound value.
    PredData = PredData.clip(max=255, min=0)     # ASUS-Joseph-18082201

    # ASUS-Joseph-18082202 >>>
    ###  Predict original image
    dataY = []

    if DENOISE_USING_THREADS:
        ### Threading method
        for j in range(int(Y_Len / DenoiseTNum)): ### create 16 threads
            thread = []
            for k in range(DenoiseTNum):
                thread.append(ThreadWithReturnValue(target=ThreadingDenoise, args=(PredData[(j*DenoiseTNum+k)*X_Len:((j+1)*DenoiseTNum+k)*X_Len], k, )))
                #print("PredData[%d:%d] is assigned to thread %d" %((j*DenoiseTNum+k)*X_Len, ((j+1)*DenoiseTNum+k)*X_Len, k))
                thread[k].start()

            for k in range(DenoiseTNum):
                dataY.append(thread[k].join())
    else:
        ### serial method
        for j in range(Y_Len):
            dataX = []
            for i in range(X_Len):
                Seg = np.reshape(PredData[i + j * X_Len], (SHei, SWid, 3))
                Seg = DenoiseImgArray(Seg)
                dataX.append(Seg)

            dataX = np.concatenate((dataX), axis=1)
            dataY.append(dataX)

    
    PredData = np.asarray(dataY)
    PredData = np.reshape(PredData, (3456, 4608, 3))

    ### Save the image before blend (post processing)
    
    PredFile = flist[TestFileIndex].rsplit('.', 1)[0] + "_pred_bfBlend" +  str(findex) + ".jpg"
    print("The predicted image without blend: %s is produced" %(PredFile))
    Cim = Image.fromarray(np.asarray(np.around(PredData), dtype=np.uint8), mode="RGB")  # floating number around rule follows IEEE
    Cim.save(PredFile, quality = 100)
    ShowImage(PredFile)
    

    ### Predict shifted image
    PredData = BlendShiftedPredData(model_1, PredData, flist[TestFileIndex])
    PredFile = flist[TestFileIndex].rsplit('.', 1)[0] + "_pred_" +  str(findex) + ".jpg"
    print("The predicted image with blend: %s is produced" %(PredFile))
    Cim = Image.fromarray(np.asarray(np.around(PredData), dtype=np.uint8), mode="RGB")
    Cim.save(PredFile, quality = 100)
    ShowImage(PredFile)
