##########################
#
#   Global variables - 
#   token declaration
#
##########################

### Mode Control
global TRANING_MODE_ENABLE
global TESTING_IMAGE_ENABLE
global DENOISE_USING_THREAD

### Save or restore model
global AUTO_SAVE_MODEL_ENABLE
global FINISH_SAVE_MODEL_ENABLE
global RESTORE_MODEL_ENABLE

### Training parameter
global  SWid
global  SHei
global  FILE_LOAD
global BATCH_SIZE
global NumImgTrainedB4Test
global DenoiseTNum      # threads number of predicted image denoise process

### Store and resotre model parameter
global EXPORT_PATH
global IMPORT_PATH

##########################
#
#   MODES selection
#
##########################

### condition 1: Restore model to predict images: 
#   TRANING_MODE_ENABLE = 0 , RESTORE_MODEL_ENABLE = 1
### condition 2: Train model with predict image and save model in every NumImgTrainedB4Test photo
#   TRANING_MODE_ENABLE = 1, TESTING_IMAGE_ENABLE = 1, AUTO_SAVE_MODEL_ENABLE = 1

TRANING_MODE_ENABLE = 1     # Training a new model
TESTING_IMAGE_ENABLE = 1    # predict images and save to disk

### Render 
DENOISE_ENABLE = 0 & TESTING_IMAGE_ENABLE
BLEND_ENABLE = 0 & TESTING_IMAGE_ENABLE
DENOISE_USING_THREADS = 0 & DENOISE_ENABLE     # use threads or serial computing to denoise predicted image

### Save and restore model
AUTO_SAVE_MODEL_ENABLE  =   1 & TRANING_MODE_ENABLE
ASK_FOR_SAVE_MODEL_ENABLE = 1 & AUTO_SAVE_MODEL_ENABLE      # Can decide whether to save the model after review the current predicted image
FINISH_SAVE_MODEL_ENABLE =  0 & TRANING_MODE_ENABLE

RESTORE_MODEL_ENABLE = 0    


##########################
#
#   Control variables
#
##########################


## Slice a image into SWid * SHei pieces of image
FILE_LOAD = 7  # number of cropped image loaded into memory at once (consider memory size limitation 5 to 10 is a proper range)
SHei = 18
SWid = 18
BATCH_SIZE = 250

if TESTING_IMAGE_ENABLE | AUTO_SAVE_MODEL_ENABLE:
    NumImgTrainedB4Test = 35

if TESTING_IMAGE_ENABLE & DENOISE_USING_THREADS:
    DenoiseTNum = 8

if AUTO_SAVE_MODEL_ENABLE | FINISH_SAVE_MODEL_ENABLE:
    EXPORT_PATH = "./_model\MiniDnn_18x18_500_200_100/model_1.ckpt"

if RESTORE_MODEL_ENABLE:
    IMPORT_PATH = "./_model/MiniDnn_18x18_500_200_100/model_1.ckpt"