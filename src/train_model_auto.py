'''
a pipeline that trains flood models automatically

this code has a main entry and was runned through the terminal

this code may not working on other computers as it depends on where the data are stored and how they are named.

but the general idea of this code is:

0. start the program with a list of user-given arguments (i.e., running mode, dataset selection, txt file location)
1. load a list of json configuration files based on the txt, and process them one after the other, each json file defines a training task
2. for each json file, start a sub-process
3. load training data, and do data processing as specified in the json file (such as shuffling traing / test data, computing features, sampling patches from the catchment data) 
4. build CNN model / load pre-trained model according to the running mode
5. training the CNN model / run validation tasks according to the running mode
6. save the model / save the validation results
7. close the sub-process so that everything is removed from the memory, then move to the next json

I used sub-process because tensorflow (or maybe keras) has no easy way to clean up everything and start over

the txt and the json files are given as examples to help go through the code
'''
import fe
import load_data
import unet
import validation_util

import numpy as np
import psutil
import os
import math
import random
import json
#import tensorflow.keras as keras # causing problems when using data generator
import keras
from tensorflow.keras import backend as K
import tensorflow as tf

DEFAULT_DATA_SOURCE='zurich'
TYPE_NAME='patch'

def pad_array(nd_array, patch_size, pad_val=0, extra_pad_size=0, return_pad_size=False):
    '''
    pad a nd-array with pad_val so that the result is large enough for at least one patch_size x patch_size patch
    
    nd_array: the original array
    patch_size: patch size
    pad_val: value used for padding
    extra_pad_size: extra padding size so that the has-data pixels at the boundary will locate in the center of a result patch
    '''
    h2 = nd_array.shape[0]
    w2 = nd_array.shape[1]

    pad_h = max(max(0, 1 + (patch_size - h2) // 2), extra_pad_size)
    pad_w = max(max(0, 1 + (patch_size - w2) // 2), extra_pad_size)
    
    if pad_h > 0 or pad_w > 0:
        #print (nd_array.shape)
        if len(nd_array.shape)==2:
            nd_array = np.pad(nd_array, ((pad_h, pad_h), (pad_w, pad_w)), "constant", constant_values=pad_val)
        else:
            # pad only x and y dimensions
            nd_array = np.pad(nd_array, ((pad_h, pad_h), (pad_w, pad_w),(0,0)), "constant", constant_values=pad_val)
#         print ("\t pad to", nd_array.shape)

    if return_pad_size:
        return nd_array,pad_h,pad_w
    else:
        return nd_array

def get_catchment_root(conf):
    return os.path.join(conf['data_root'],
                        conf['data_resolution']+'_'+conf['data_features'],
                        get_data_source(conf))

def get_data_source(conf):
    return DEFAULT_DATA_SOURCE if not 'optional_data_source' in conf.keys() else conf['optional_data_source']
    
def get_catchment_names(catchment_root, seed=1, propotion=3, load_num=None, train_test=True):
    '''
    get the list of catchment areas from given folder, split them into training / test sets
    
    catchment_root: root folder
    seed: random seed
    propotion: 1 out of propotion of catchments will be test set, the rests will be training set
    load_num: a debug option, how many catchments in total will be read
    '''
    random.seed(seed)
    
    catchments = [f.split('_')[0] for f in os.listdir(catchment_root)]
    random.shuffle(catchments)

    if load_num is not None:
        catchments = catchments[:load_num]

    if train_test:
        catchments_train = catchments[len(catchments)//propotion:]
        catchments_test = catchments[:len(catchments)//propotion]

        return catchments_train,catchments_test
    else:
        return catchments

def expand_channel(arr, c, fe):
    '''
    replace channel c of the input nd-array arr by the result of feature extractor fe
    '''
    in_channels=arr.shape[-1]
    c_target = fe.channels()
    out_channels=in_channels+c_target-1
    
    if in_channels==out_channels: # c_target==1
        arr[...,c:c+1]=fe.features(arr[...,c])
        return arr
    else:
        arr2 = fe.features(arr[...,c])
        #print(arr2.shape)
        arr_list=[]
        if c > 0:
            arr_list.append(arr[...,:c])
        arr_list.append(arr2)
        if c < in_channels-1:
            arr_list.append(arr[...,c+1:])
        return np.concatenate(arr_list,axis=-1)

# configureation
def default_config(file):
    '''
    generate a json configuration file.
    
    note that the settings are default values and should be manually changed
    '''
    config={'type':TYPE_NAME,
            'name':'default',
            'net_filter':[32,64],
            'net_kernel':[[3,3],[3,3]],
            'net_stride':[[1,1],[1,1]],
            'net_u_connections':True,
            'net_height_in_prediction':False, # when set to positive float (called val), the prediction will be (wd+ele)*val (not working, tested already)
            'data_root':'give path here',
            'data_resolution':'give string here',
            'data_features':'give string here',
            'data_patch_size':256,
            'data_random_seed':1,
            'data_propotion':3,
            'data_output_channels':2,
            'data_patch_ratio':2,
            'data_augmentation':True, # data augmentation for training
            'data_aug_rot':60, # False: no rotation, integer: with rotation
            'data_aug_flip_x':[-1,1], # x flip
            'data_aug_flip_y':[1], # y flip
            'data_feature_extractor':0, # integer, type of feature extractor to use
            'train_batch_size':2,
            'train_epoch':200,
            'train_learning_rate':0.00005,
            'train_optimizer':'adam',
            'train_weighted_loss_s':False, # set a number to activate weighted loss
            'train_weighted_loss_c':False # set a number to activate weighted loss
           }
    with open(file,'w',encoding='utf-8') as f:
        json.dump(config,f)

def load_training_data(conf, fe_x, train=True, test=True, debug=False):
    # training/test sets
    catchment_root=get_catchment_root(conf)
    features=conf['data_features']
    
    catchments_train,catchments_test = get_catchment_names(catchment_root,
                                                           conf['data_random_seed'],
                                                           conf['data_propotion']
                                                          )
    
    if 'optional_mix_data' in conf.keys():
        if conf['optional_mix_data']:
            catchments_all=catchments_train+catchments_test
            catchments_train=catchments_all
            catchments_test=catchments_all
    
    if debug:
        catchments_train=catchments_train[:10]
        catchments_test=catchments_test[:10]
    
    output_channels=conf['data_output_channels']
    patch_size=conf['data_patch_size']
    
    # output size==input size??
    output_size_offset=0
    if 'data_output_size_offset' in conf.keys(): # output patch is smaller (central part of the input patch)
        output_size_offset=conf['data_output_size_offset']
    # output size used for patch sampling
    patch_size_sampling=patch_size-output_size_offset-output_size_offset
    
    fe.DEBUG=False # dont show the debug message

    if train:
        # nd-images and training patches
        print ('loading nd-array training set to memory')
        nd_img_train = [pad_array(expand_channel(np.load(os.path.join(catchment_root, c + '_' + features + '.npy')), 0, fe_x),
                                  patch_size_sampling, 0, 0) for c in catchments_train]
        print('memory % used:', psutil.virtual_memory()[2])
        # print debug information regarding data preprocessing
    if test:
        print ('loading nd-array test set to memory')
        nd_img_test = [pad_array(expand_channel(np.load(os.path.join(catchment_root, c + '_' + features + '.npy')), 0, fe_x),
                                  patch_size_sampling, 0, 0) for c in catchments_test]
        print('memory % used:', psutil.virtual_memory()[2])
        
    print ('generate patches')

    # the default patch generator, uniform random
    patch_generator=load_data.uniform_patch_location_generator(patch_size_sampling,Ellipsis,0,None,ratio=conf['data_patch_ratio'])
    # if data_patch_sampling is specified
    if 'data_patch_sampling' in conf.keys():
        if conf['data_patch_sampling']=='inverse_mean':
            # a rough weighted sampling here, a is smaller for small patch
            
            # with a=10
            # x.mean() | prob
            # 0.01 | 0.095
            # 0.1 | 0.61
            # 0.2 | 0.84
            # 0.3 | 0.93
            # 0.4 | 0.97

            # with a=5
            # x.mean() | prob
            # 0.01 | 0.049
            # 0.1 | 0.38
            # 0.2 | 0.60
            # 0.3 | 0.73
            # 0.4 | 0.81
            
            a=10/((1024/patch_size_sampling)**0.5)
            print('patch_generator','inverse_mean, a=',a)
            patch_generator=load_data.weighted_patch_location_generator(patch_size_sampling,Ellipsis,0,None,
                                                                        ratio=conf['data_patch_ratio'],
                                                                        weight_func=load_data._prob_inverse_x(a=a),
                                                                        weight_channel=-2)
            
        elif conf['data_patch_sampling']=='wheel':
            # patch-sampling based on wheel selection of center pixel
            patch_generator=load_data.wheel_weighted_patch_location_generator(patch_size_sampling,Ellipsis,0,None,
                                                                              ratio=conf['data_patch_ratio'])

    patch_regen=False
    if 'data_patch_regen_per_epoch' in conf.keys():
        patch_regen=conf['data_patch_regen_per_epoch']
        
    if train:
        if conf['data_aug_rot']:
            if output_size_offset:
                raise Exception('rotation does not coexist with output_size_offset')
                
            patch_train = load_data.PatchFromImageWithRotation(nd_img_train,patch_size,
                                                               output_channels=output_channels,
                                                               ratio=conf['data_patch_ratio'],
                                                               batch_size=conf['train_batch_size'],
                                                               augmentation=conf['data_augmentation'],
                                                               rot_step=conf['data_aug_rot'],
                                                               flip_x=conf['data_aug_flip_x'],
                                                               flip_y=conf['data_aug_flip_y'],
                                                               patch_loc_generator=patch_generator,
                                                               regen_patch_every_epoch=patch_regen)
        else:
            patch_train = load_data.PatchFromImage(nd_img_train, patch_size,
                                                   output_channels=output_channels,
                                                   ratio=conf['data_patch_ratio'],
                                                   batch_size=conf['train_batch_size'],
                                                   augmentation=conf['data_augmentation'],
                                                   patch_loc_generator=patch_generator,
                                                   output_size_offset=output_size_offset,
                                                   regen_patch_every_epoch=patch_regen)
    else:
        patch_train=None
        
    if test:
        patch_test = load_data.PatchFromImage(nd_img_test,patch_size,
                                              output_channels=output_channels,
                                              ratio=conf['data_patch_ratio'],
                                              batch_size=conf['train_batch_size'],
                                              shuffle=False,
                                              output_size_offset=output_size_offset)
    else:
        patch_test=None
    return patch_train, patch_test

def feature_extractor(conf):
    '''
    this controls the feature extractor
    '''
    if conf['data_feature_extractor']==1:
        print('use feature extractor 1')
        return fe.Features([fe.TranslatedHeight(scale=0.001),fe.GeographicByDilate(1, keep_channels=[0,1,4])])
    elif conf['data_feature_extractor']==2:
        print('use feature extractor 2')
        return fe.Features([fe.ReverseTranslatedHeight(scale=0.01),fe.GeographicByDilate(1, keep_channels=[0,1,4])])
    elif conf['data_feature_extractor']==3:
        print('use feature extractor 3')
        return fe.Features([fe.ReverseTranslatedHeight(scale=0.01),fe.Mask()])
    # the default one
    return fe.Features([fe.NormalizedHeight([]), fe.GeographicByDilate(1, keep_channels=[0,1,4])])
        
def clear_tf():
    # https://stackoverflow.com/questions/58453793/the-clear-session-method-of-keras-backend-does-not-clean-up-the-fitting-data
    K.clear_session()
    tf.reset_default_graph()

def train(model, conf, patch_train, patch_test):
    model_name = conf['name']
    model_folder = os.path.join("trained_models",TYPE_NAME, model_name)
    
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    # training
    cb_chkpt=keras.callbacks.ModelCheckpoint(os.path.join(model_folder, "chkpt_best.h5"), monitor='val_loss', save_best_only=True, mode='min', period=1)
    cb_hist=keras.callbacks.CSVLogger(os.path.join(model_folder, "loss.log"), separator=',', append=True)

    model.fit_generator(patch_train, validation_data=patch_test, epochs=conf['train_epoch'], callbacks=[cb_chkpt, cb_hist], workers=1)
    model.save(os.path.join(model_folder, "chkpt_last.h5"))

    
def evaluate(model, conf, patch_train, patch_test):
    model_name = conf['name']
    model_folder = os.path.join("trained_models",TYPE_NAME, model_name)
    
    loss_value_train=None
    if patch_train is not None:
        loss_value_train=model.evaluate_generator(patch_train, workers=1)
    
    loss_value_test=None
    if patch_test is not None:
        loss_value_test=model.evaluate_generator(patch_test, workers=1)

    print('losses', loss_value_train, loss_value_test)
    with open(os.path.join(model_folder, "evaluate.txt"),'w') as f:
        f.write(str(loss_value_train)+','+str(loss_value_test))

def load_pre_trained_model(conf):
    path = os.path.join("trained_models",TYPE_NAME)
    return unet.load_pre_trained_model(path,conf)

# rewrite functions, there are only two test function, test_by_validation_data and test_by_catchment
def test_by_validation_data(model, conf, patch_test, data_name='validation_set'):
    '''
    test the model by test set (patch data that used for training and testing)
    '''
    max_wd_safe=20 # m
    max_vel_safe=20 # m/s
    res_wd=20 # 0.05m
    res_vel=20 # 0.05m

    data_source=get_data_source(conf)

    validation_util.validation_with_test_set(model,conf,patch_test,
                                             matrix_max=[max_wd_safe,max_vel_safe],
                                             matrix_res=[res_wd,res_vel],
                                             output_channel_name=['wd','vel'],
                                             data_name=os.path.join(data_source,data_name)
                                            )
import time

def test_by_catchment(model, conf, fe_x):
    '''
    test the model by all catchments from data_source folder (complete dem input)
    
    note that
    1.the complete folder is conf['data_root']/conf['data_resolution']+'_'+conf['data_features']/data_source
    2.for every data_source the result json file will specify which catchment area is training set and which is test set, this is based on random shuffle of filenames and it is only valid if the model is trained using the same data source
    '''
    data_source=get_data_source(conf)
    test_folder = os.path.join("tests",conf['type'], conf['name'], data_source)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    catchment_root=get_catchment_root(conf)
    
    catchments_train,catchments_test=get_catchment_names(catchment_root,conf['data_random_seed'],conf['data_propotion'])
    
    if 'optional_training_test_selection' in conf.keys():
        if conf['optional_training_test_selection']==0:
            all_catchments=catchments_train+catchments_test
            print('use train + test')
        elif conf['optional_training_test_selection']==1:
            all_catchments=catchments_train
            print('use train')
        else:
            all_catchments=catchments_test
            print('use test')

    else:
        # by default using only test set
        if 'optional_data_source' not in conf.keys():
            all_catchments=catchments_test
            print('use test')
        else:
            # if data source is specified, use train + test set when working on non-default data source
            if conf['optional_data_source']==DEFAULT_DATA_SOURCE:
                all_catchments=catchments_test
                print('use test')
            else:
                all_catchments=catchments_train+catchments_test
                print('use train + test')
        
    features=conf['data_features']
    input_channels=fe_x.channels()
    output_channels=conf['data_output_channels']
    patch_size=conf['data_patch_size']
    
    print ('testing, data source:',data_source)
    all_info={}
    for c in all_catchments:
        inf={}
        # load array, preprocess
        nd_array_raw=np.load(os.path.join(catchment_root, c + '_' + features + '.npy'))
        
        dt=time.time()
        nd_array,pad_h,pad_w=pad_array(expand_channel(nd_array_raw, 0, fe_x),patch_size, 0, 0, return_pad_size=True)
        dt=time.time()-dt
        inf['time_preprocess']=dt
        
        dt=time.time()
        # make prediction, record mse
        y_truth=nd_array[...,-output_channels:]
        y_pred, patch_num=validation_util.predict_catchment_patch(model,nd_array,output_channels,
                                                                  patch_size,return_patch_num=True)
        
        dt=time.time()-dt
        inf['time_predict']=dt
        inf['patch_num']=patch_num
        
        # exclude no-data pixels
        mask=np.logical_not(np.all((nd_array[...,:input_channels]==0),axis=-1))
        
        inf['MAE']=float(np.square(y_truth-y_pred)[mask].mean())
        inf['test_set']=(c in catchments_test)
        all_info[c]=inf
        
        # unpad to original size
        y_pred=y_pred[pad_h:y_pred.shape[0]-pad_h,pad_w:y_pred.shape[1]-pad_w]
        print(nd_array_raw.shape, y_pred.shape)
        # save result
        np.save(os.path.join(test_folder, c + '.npy'), y_pred)
        
    with open(os.path.join(test_folder, 'all_info.json'),'w') as f:
        json.dump(all_info,f)

# use a sub-process to run the model training
# see https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/96876
# seems bug exitis in keras, and clear_session() dont clear everything
import multiprocessing

def train_wrapper(conf):
    print(__name__)
    
    fe_x=feature_extractor(conf)
    model=unet.build(conf, fe_x.channels())
    patch_train, patch_test=load_training_data(conf, fe_x)
    train(model, conf, patch_train, patch_test)
    test_by_validation_data(model, conf, patch_test)

    # delete everything, get ready for the next one
    del patch_train
    del patch_test
    del model
    clear_tf()
    
def continue_train_wrapper(conf):
    print(__name__)
    
    fe_x=feature_extractor(conf)
    
    model_name = conf['name']
    model_folder = os.path.join("trained_models",TYPE_NAME, model_name)
    
    if not os.path.exists(os.path.join(model_folder, "chkpt_last.h5")):
        print("model not exist, create a new model for training")
        model=unet.build(conf, fe_x.channels())
    else: 
        print("loading pre-trained model")
        model=load_pre_trained_model(conf)
        
    patch_train, patch_test=load_training_data(conf, fe_x)
    train(model, conf, patch_train, patch_test)
    test_by_validation_data(model, conf, patch_test)

    # delete everything, get ready for the next one
    del patch_train
    del patch_test
    del model
    clear_tf()
    
def evaluate_wrapper(conf):
    print(__name__)
    fe_x=feature_extractor(conf)
    model=load_pre_trained_model(conf)

    if not 'optional_training_test_selection' in conf.keys():
        patch_train, patch_test=load_training_data(conf,fe_x,train=False)
    else:
        if conf['optional_training_test_selection']==0:
            patch_train, patch_test=load_training_data(conf,fe_x)
        elif conf['optional_training_test_selection']==1:
            patch_train, patch_test=load_training_data(conf,fe_x,test=False)
        else:
            patch_train, patch_test=load_training_data(conf,fe_x,train=False)
    
    evaluate(model, conf, patch_train, patch_test)
    
def test_wrapper(conf):
    print(__name__)

    fe_x=feature_extractor(conf)
    model=load_pre_trained_model(conf)
    
    if not 'optional_training_test_selection' in conf.keys():
        _, patch_test=load_training_data(conf,fe_x,train=False)

        # test_by_validation_data(model, conf, patch_train,data_name='training_set')
        test_by_validation_data(model, conf, patch_test,data_name='validation_set')
    else:
        if conf['optional_training_test_selection']==0:
            patch_train, patch_test=load_training_data(conf,fe_x)

            test_by_validation_data(model, conf, patch_train,data_name='training_set')
            test_by_validation_data(model, conf, patch_test,data_name='validation_set')
        elif conf['optional_training_test_selection']==1:
            patch_train, _=load_training_data(conf,fe_x,test=False)
            test_by_validation_data(model, conf, patch_train,data_name='training_set')
        else:
            _, patch_test=load_training_data(conf,fe_x,train=False)
            # test_by_validation_data(model, conf, patch_train,data_name='training_set')
            test_by_validation_data(model, conf, patch_test,data_name='validation_set')

    del model
    clear_tf()

def test_catchment_wrapper(conf):
    print(__name__)
    
    fe_x=feature_extractor(conf)
    model=load_pre_trained_model(conf)
    test_by_catchment(model, conf, fe_x)
    
    # delete everything,  get ready for the next one
    del model
    clear_tf()
        
def run_on_confs(list_of_files,mode,options):
    '''
    train on a list of conf files
    '''
    for file in list_of_files:
        with open(file,encoding='utf-8') as f:
            conf=json.load(f)
            
        if conf['type']!=TYPE_NAME:
            print('invalid configuration file, exit!')
            continue
        
        # copy options to conf so that they can be used
        for k in options.keys():
            conf[k]=options[k]
            
        if mode=='-t':
            p = multiprocessing.Process(target=train_wrapper, args=(conf,))
        elif mode=='-ct':
            p = multiprocessing.Process(target=continue_train_wrapper, args=(conf,))
        elif mode=='-vc':
            p = multiprocessing.Process(target=test_catchment_wrapper, args=(conf,))
        elif mode=='-e':
            p = multiprocessing.Process(target=evaluate_wrapper, args=(conf,))
        else:
            p = multiprocessing.Process(target=test_wrapper, args=(conf,))
            
        p.start() # start the process
        p.join() # wait the process finishes (we dont need parallel processing, we use sub process to handle the memory issue of keras)

if __name__=='__main__':
    import sys
    args = sys.argv
    if len(args)<3:
        print('usage:\n1. modes (only one should be used at the same time)\n',
              'save default config json file: -c filename\n',
              'fresh training using a list of config files: -t txtfilename\n',
              'continue training using a list of config files: -ct txtfilename\n',
              'testing on a list of config files: -v txtfilename\n',
              'testing on a list of config files, by each catchment area: -vc txtfilename\n',
              '2. options\n',
              'specify another data source: -ds foldername\n',
              'training/test set selection for mode -v and -vc (0 both, 1 training set, 2 test set): -tts value\n',
              'the default data source is zurich\n',
              'data sources should be placed at [data_root]/[data_resolution]_[data_features]/\n',
              'vairables surrounded by [] should be specified in the json configuration file\n',
              'for training and testing, a txt file in which each line is the name of a config json file')
    else:
        args=args[1:]
        
        mode='-v'
        list_of_confs=None
        options={}
        
        for i in range(0,len(args),2):
            if args[i]=='-c':
                default_config(args[i+1])
            elif args[i]=='-t' or args[i]=='-v' or args[i]=='-vc' or args[i]=='-ct' or args[i]=='-e':
                mode=args[i]
                
                with open(args[i+1]) as f:
                    list_of_confs=f.readlines()
                list_of_confs = [x.strip() for x in list_of_confs]
            elif args[i]=='-ds':
                options['optional_data_source']=args[i+1]
            elif args[i]=='-tts':
                options['optional_training_test_selection']=int(args[i+1])
            elif args[i]=='-m':
                # add training set to test set (so that they share the same data augmentation methods, for validation set only)
                options['optional_mix_data']=int(args[i+1])
            else:
                print('cannot understand argument',args[i])
            
        run_on_confs(list_of_confs,mode,options)
