import os
import random

FILENAME_FILTER_REAL_ELEVATION=lambda x:True
FILENAME_FILTER_SYN_ELEVATION=lambda x:'-max' in x

OUTPUT_FILTER_REAL_ELEVATION=lambda x:x.split('_')[0]
OUTPUT_FILTER_SYN_ELEVATION=lambda x:x

FILENAME_FILTER_UNCHANGE=lambda x:True
OUTPUT_FILTER_UNCHANGE=lambda x:x

def get_catchment_names(catchment_root,
                        seed=1,
                        propotion=3,
                        filename_filter=FILENAME_FILTER_UNCHANGE,
                        output_filter=OUTPUT_FILTER_UNCHANGE,
                        load_num=None,
                        train_test=True):
    '''
    get the list of catchment areas from given folder, split them into training / test sets
    each catchment area is named as id + '_' + features + '.npy'
    the function only returns id
    
    catchment_root: root folder
    seed: random seed
    propotion: 1 out of propotion of catchments will be test set, the rests will be training set
    load_num: a debug option, how many catchments in total will be read
    train_test: whether to split the training/test set
    '''
    random.seed(seed)
    
    catchments = [output_filter(f) for f in os.listdir(catchment_root) if filename_filter(f)]
    random.shuffle(catchments)

    if load_num is not None:
        catchments = catchments[:load_num]

    if train_test:
        catchments_train = catchments[len(catchments)//propotion:]
        catchments_test = catchments[:len(catchments)//propotion]

        return catchments_train,catchments_test
    else:
        return catchments

# for testing
# a,b=get_catchment_names('E:\\training_patches\\zurich\\4m_dem_wd-max_vel-max',output_filter=OUTPUT_FILTER_REAL_ELEVATION)
# print(a)
# print(b)