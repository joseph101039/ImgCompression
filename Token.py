##########################
#
#   Global variables - 
#   Token declaration
#
##########################

### Mode Control
global TRANING_MODE_ENABLE
global TESTING_IMAGE_ENABLE
global DENOISE_USING_THREAD

### Training parameter
global  SWid
global  SHei
global  FILE_LOAD
global BATCH_SIZE
global NumImgTrainedB4Test
global DenoiseTNum      # threads number of predicted image denoise process

##########################
#
#   MODES selection
#
##########################

TRANING_MODE_ENABLE = 1     # Training a new model
TESTING_IMAGE_ENABLE = 1    # predict images and save to disk
DENOISE_USING_THREADS = 0    # use threads or serial computing to denoise predicted image

##########################
#
#   Control variables
#
##########################

if TRANING_MODE_ENABLE:
    ## Slice a image into SWid * SHei pieces of image
    FILE_LOAD = 50  # number of cropped image loaded into memory at once (consider memory size limitation 5 to 10 is a proper range)
    SHei = 18
    SWid = 18
    BATCH_SIZE = 250
if TESTING_IMAGE_ENABLE:
    NumImgTrainedB4Test = 10

if TESTING_IMAGE_ENABLE & DENOISE_USING_THREADS:
    DenoiseTNum = 5
