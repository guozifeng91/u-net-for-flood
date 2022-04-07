import keras
# import tensorflow as tf
from tensorflow.keras import backend as K
import os

'''
calculate how much cells of input layer can be visited by one cell of the last layer, for conv network (receptive field)

k is the list of kernel sizes
s is the list of stride sizes

1 denote the output of the 1st conv layer

therefore we have:
r(1) = k(1)
r(n) = r(n-1) + (k(n)-1) * s(n-1) * ... * s(1)
'''
def r(k, s, i=-1):
    if i==-1:
        i=len(k)-1
        
    if i==0:
        return k[i]
    else:
        dot=1
        for j in range(i):
            dot = dot*s[j]
        return r(k,s,i-1) + (k[i]-1)*dot

class weighted_loss():
    def __init__(self, s, c):
        self.s=s
        self.c=c
    def loss_func(self, y_true, y_pred):
        # use the wd channel as weight?
        return K.mean(K.square(y_pred-y_true) * K.exp(y_true*self.s+self.c), axis=-1)
        
class masked_loss():
    def __init__(self, s=2, c=0.0001):
        self.s=s
        self.c=c
    def loss_func(self, y_true, y_pred):
        # use tanh to binarize truth, then use it as mask for the loss
        mask=(K.tanh(K.abs(y_true*self.s))+self.c)
        return K.sum(K.square(y_pred-y_true)*mask)/(K.sum(mask)+0.001) # plus 0.001 to prevent divide by 0
        #return K.mean(K.square(y_pred-y_true) * (K.tanh(K.abs(y_true*self.s))+self.c), axis=-1)

class masked_loss_2():
    def __init__(self,output_channels,mask_channel=-1):
        self.output_channels=output_channels
        self.mask_channel=mask_channel
        
    def loss_func(self, y_true, y_pred):
        print(y_true)
        print(y_pred)
        mask=K.repeat_elements(K.cast(y_pred[...,self.mask_channel:]>0,'float32'),rep=self.output_channels,axis=-1)
        print(mask)
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/boolean_mask
        #return K.mean(K.boolean_mask(K.square(y_pred-y_true), mask))
        return K.sum(K.square(y_pred[...,:self.output_channels]-y_true)*mask)/K.sum(mask)
        
def load_pre_trained_model(path,conf):
    K.clear_session()
    
    model_name = conf['name']
    model_folder = os.path.join(path, model_name)
    
    model=None
    
    if 'train_weighted_loss_s' in conf.keys() and 'train_weighted_loss_c' in conf.keys():
        if conf['train_weighted_loss_s'] and conf['train_weighted_loss_c']:
            model=keras.models.load_model(os.path.join(model_folder, "chkpt_last.h5"),
                                          custom_objects={'loss_func': loss_func(conf)})
            
    if 'train_masked_loss_s' in conf.keys() and 'train_masked_loss_c' in conf.keys():
        if conf['train_masked_loss_s'] and conf['train_masked_loss_c']:
            model=keras.models.load_model(os.path.join(model_folder, "chkpt_last.h5"),
                                          custom_objects={'loss_func': loss_func(conf)})
                                          
    if 'train_include_mask_in_output' in conf.keys():
        if conf['train_include_mask_in_output']:
            model=keras.models.load_model(os.path.join(model_folder, "chkpt_last.h5"),
                                          custom_objects={'loss_func': loss_func(conf)})                      
    if model is None:
        # use mse
        model=keras.models.load_model(os.path.join(model_folder, "chkpt_last.h5"))
    # model.summary()
    return model
    
def loss_func(conf):
    '''
    return a loss function based on the conf
    '''
    if 'train_weighted_loss_s' in conf.keys() and 'train_weighted_loss_c' in conf.keys():
        if conf['train_weighted_loss_s'] and conf['train_weighted_loss_c']:
            print('use weighted loss')
            return weighted_loss(conf['train_weighted_loss_s'],conf['train_weighted_loss_c']).loss_func
            
    if 'train_masked_loss_s' in conf.keys() and 'train_masked_loss_c' in conf.keys():
        if conf['train_masked_loss_s'] and conf['train_masked_loss_c']:
            print('use m loss')
            return masked_loss(conf['train_masked_loss_s'],conf['train_masked_loss_c']).loss_func

    if 'train_include_mask_in_output' in conf.keys():
        mask_channel=-1 # always use -1 channel
        if conf['train_include_mask_in_output']:
            print('use m loss with m specified in output')
            return masked_loss_2(conf['data_output_channels'],mask_channel).loss_func
            
    return 'mean_squared_error'

def encoder(inputs, filters, kernel_size, stride, kernel_init, bias_init):
    '''
    sequence of conv layers
    return final layer, list of layers, and filters
    '''
    layer = inputs
    length = len(filters)
    
    if not hasattr(filters, '__len__'):
        filters = [filters]
    
    if not hasattr(kernel_size, '__len__'):
        kernel_size = [(kernel_size,kernel_size)] * len(filters)
    else:
        if not hasattr(kernel_size[0], '__len__'):
            kernel_size =  [kernel_size] * len(filters)
         
    if not hasattr(stride, '__len__'):
        stride = [(stride,stride)] * len(filters)
    else:
        if not hasattr(stride[0], '__len__'):
            stride = [stride] * len(filters)
            
    conv_list=[]
    
    for i in range(length):
        layer = keras.layers.Conv2D(filters[i], kernel_size[i], strides = stride[i], padding='same',kernel_initializer=kernel_init, bias_initializer=bias_init)(layer)
        # layer = keras.layers.BatchNormalization()(layer)
        layer = keras.layers.LeakyReLU(alpha=0.1)(layer)
        print(layer)
            
        conv_list.append(layer)
    return layer, conv_list, filters

def decoder(inputs, filters, kernel_size, stride, output_padding, kernel_init, bias_init, conv_list=None):
    '''
    convt + conv, in a reversed order as encoder
    
    note that the filter, kernel_size, stride are REVERSED as encoder
    '''
    layer = inputs
    length = len(filters)
    
    if not hasattr(filters, '__len__'):
        filters = [filters]
    
    if not hasattr(kernel_size, '__len__'): # integer
        kernel_size = [(kernel_size,kernel_size)] * len(filters)
    else: # list of integer
        if not hasattr(kernel_size[0], '__len__'):
            kernel_size =  [kernel_size] * len(filters)
         
    if not hasattr(stride, '__len__'):
        stride = [(stride,stride)] * len(filters)
    else:
        if not hasattr(stride[0], '__len__'):
            stride = [stride] * len(filters)
            
    if not hasattr(output_padding, '__len__'):
        output_padding = [(output_padding,output_padding)] * len(filters)
    else:
        if not hasattr(output_padding[0], '__len__'):
            output_padding =  [output_padding] * len(filters)
            
    for i in range(length):
        # convt as up sampling
        layer = keras.layers.Conv2DTranspose(filters[i], [2,2], strides = stride[i], output_padding=None, padding='same',kernel_initializer=kernel_init, bias_initializer=bias_init)(layer)
        # layer = keras.layers.BatchNormalization()(layer)
        layer = keras.layers.LeakyReLU(alpha=0.1)(layer)
        print(layer)

        if conv_list is not None:
            if conv_list[i] is not None: # use None to replace the element in the list where a skip connection is not wanted
                layer = keras.layers.Concatenate(axis=-1)([layer, conv_list[i]])
                print(layer)
        
        # conv continuous the process (stride = 1)
        layer = keras.layers.Conv2D(filters[i], kernel_size[i], padding='same',kernel_initializer=kernel_init, bias_initializer=bias_init)(layer)
        # layer = keras.layers.BatchNormalization()(layer)
        layer = keras.layers.LeakyReLU(alpha=0.1)(layer)
        print(layer)
        
    return layer

def encoder_segnet(inputs, filters, kernel_size, stride, pool_pos, kernel_init, bias_init, activation_for_all=True):
    '''
    sequence of conv layers + max pooling
    
    e.g.,
    filters=[8,8,8,16,16,16]
    kernel_size=[5,5,2,5,5,2]
    stride=[1,1,2,1,1,2]
    pool_pos=3
    
    will give a conv[5x5x?x8], conv[5x5x8x8], pool[2x2], conv[5x5x8x16], conv[5x5x16x16], pool[2x2] sequence
    '''
    layer = inputs
    length = len(filters)
    
    if not hasattr(filters, '__len__'):
        filters = [filters]
    
    if not hasattr(kernel_size, '__len__'):
        kernel_size = [(kernel_size,kernel_size)] * len(filters)
    else:
        if not hasattr(kernel_size[0], '__len__'):
            kernel_size =  [kernel_size] * len(filters)
         
    if not hasattr(stride, '__len__'):
        stride = [(stride,stride)] * len(filters)
    else:
        if not hasattr(stride[0], '__len__'):
            stride = [stride] * len(filters)
    
    layer_list=[]
    
    for i in range(length):
        if i%pool_pos==pool_pos-1:
            # pooling layer
            layer=keras.layers.MaxPooling2D(kernel_size[i], strides=stride[i], padding='same')(layer)
        else:
            # conv layer
            layer = keras.layers.Conv2D(filters[i], kernel_size[i], strides = stride[i], padding='same',kernel_initializer=kernel_init, bias_initializer=bias_init)(layer)
            # layer = keras.layers.BatchNormalization()(layer)
            if activation_for_all or i%pool_pos==pool_pos-2:
                # activation after each conv (activation_for_all==true)
                # or after all conv(i%pool_pos==pool_pos-2)
                layer = keras.layers.LeakyReLU(alpha=0.1)(layer)
        print(i,layer)
            
        #if i % pool_pos==pool_pos-1:
        layer_list.append(layer)
        
    return layer, layer_list, filters

def decoder_segnet(inputs, filters, kernel_size, stride, pool_pos, kernel_init, bias_init, layer_list=None, convt=True, activation_for_all=True, only_one_conv=False):
    '''
    sequence of convT layers + conv layers
    
    e.g.,
    filters=[8,8,8,16,16,16]
    kernel_size=[5,5,2,5,5,2]
    stride=[1,1,2,1,1,2]
    pool_pos=3
    
    will give a convT[2x2x16x16], conv[5x5x16x16],  conv[5x5x16x16], convT[2x2x16x8], conv[5x5x8x8], conv[5x5x8x8] sequence
    
    the order of filters, kernel_size and stride are THE SAME with encoder_segnet
    '''
    layer = inputs
    length = len(filters)
    
    if not hasattr(filters, '__len__'):
        filters = [filters]
    
    if not hasattr(kernel_size, '__len__'): # integer
        kernel_size = [(kernel_size,kernel_size)] * len(filters)
    else: # list of integer
        if not hasattr(kernel_size[0], '__len__'):
            kernel_size =  [kernel_size] * len(filters)
         
    if not hasattr(stride, '__len__'):
        stride = [(stride,stride)] * len(filters)
    else:
        if not hasattr(stride[0], '__len__'):
            stride = [stride] * len(filters)
            
    for i in reversed(range(length)):
        # in a reversed order
        
        if i%pool_pos==pool_pos-1:
            if convt:
                # convt as upsampling
                layer = keras.layers.Conv2DTranspose(filters[i], [2,2], strides = stride[i], output_padding=None, padding='same',kernel_initializer=kernel_init, bias_initializer=bias_init)(layer)
            else:
                # upsampling 2d
                layer = keras.layers.UpSampling2D(size = stride[i])(layer)
        else:
            # conv layers
            # concatenate after (in terms of layer's sequence), or before (in terms of i) convt
            if i != length-1 and i%pool_pos==pool_pos-2 and layer_list is not None:
                if layer_list[i] is not None: # use None to replace the element in the list where a skip connection is not wanted
                    layer = keras.layers.Concatenate(axis=-1)([layer, layer_list[i]])
                    print(i, layer)
            
            if only_one_conv and i%pool_pos!=0:
                # if only_one_conv is activated, only one conv layer will be added (when i%pool_pos==0)
                # by default only_one_conv=False
                continue
                
            # conv continuous the process (stride = 1)
            layer = keras.layers.Conv2D(filters[i], kernel_size[i], padding='same',kernel_initializer=kernel_init, bias_initializer=bias_init)(layer)
            # layer = keras.layers.BatchNormalization()(layer)
            if activation_for_all or i%pool_pos==0:
                # activation after each conv (activation_for_all==true)
                # or after all conv(i%pool_pos==0)
                # note that here is convt, conv, ..., conv with a reversed order of i
                # therefore activation layer for the last layer corresponds to i==0
                layer = keras.layers.LeakyReLU(alpha=0.1)(layer)
        print(i, layer)

    return layer

def check_net_parameters(conf):
    '''
    update net parameters from any additional keys
    
    convert the short representation of kernel size, layers ect. to full representation
    '''
    if 'net_stack_repeat' in conf.keys():
        if conf['net_stack_repeat']:
            # a faster way to define segnet structure
            
            k=conf['net_kernel'] # must be an integer
            s=conf['net_stride'] # must be an integer
            f=conf['net_filter'] # must be a list
            n=conf['net_stack_repeat'] # must be a list of length=2

            # for filter f[i],
            #   if i < len(f)-1
            #     repeat conv layer of kernel_size = k and stride = 1 for n[0] times,
            #     add one max_pooling of kernel_size=s and stride=s
            # else
            #   repeat conv layer of kernel_size = k and stride = 1 for n[1] times,
            # 
            print('parsing net stack information')
             
            assert len(n)==2
            assert isinstance(k,int)
            assert isinstance(s,int)
            
            kernel_size=[]
            stride=[]
            filters=[]
            
            for i in range(len(f)):
                if i < len(f)-1:
                    # cnns
                    kernel_size=kernel_size+[[k,k]]*n[0]
                    stride=stride + [[1,1]]*n[0]
                    filters=filters + [f[i]]*n[0]

                    # max pooling
                    kernel_size=kernel_size+[[s,s]]
                    stride=stride+[[s,s]]
                    filters=filters + [f[i]]
                else:
                    # cnns
                    kernel_size=kernel_size+[[k,k]]*n[1]
                    stride=stride + [[1,1]]*n[1]
                    filters=filters + [f[i]]*n[1]
                    
            print('kernel',kernel_size)
            print('stride',stride)
            print('filters',filters)
            print('pool_pos',n[0]+1)
            
            conf['net_kernel']=kernel_size
            conf['net_stride']=stride
            conf['net_filter']=filters
            conf['net_pool_pos']=n[0]+1

def build(conf, input_channels):
    '''
    build the network
    '''
    check_net_parameters(conf)
    
    k=[i[0] for i in conf['net_kernel']]
    s=[i[0] for i in conf['net_stride']]
    
    print('receptive field', r(k,s))
    
    if 'net_pool_pos' in conf.keys():
        # use pooling layer
        if conf['net_pool_pos']:
            # use None for input shape instead of 'data-patch-size'
            # in this case, the 'data-patch-size' is used as the minimum allowed input size (latent layer size >=1)
            if 'net_non_input_shape' in conf.keys():
                if conf['net_non_input_shape']:
                    return _build_segnet_no_shape(conf, input_channels)
                
            return _build_segnet(conf, input_channels)
                
    
    # no pooling layer
    return _build(conf, input_channels)

def _add_mask_in_output(conf, input_x, output):
    # added in 2020-12-03
    if 'train_include_mask_in_output' in conf.keys():
        if conf['train_include_mask_in_output']:
            print('mask added to the output')
            which_channel_as_mask=0 # use the first channel (normalized elevation data) as mask
            mask_layer=keras.layers.Lambda(lambda x:x[...,which_channel_as_mask:which_channel_as_mask+1])
            mask_x=mask_layer(input_x)
            print(mask_x)
            
            output = keras.layers.Concatenate(axis=-1,name='mask_layer')([output, mask_x])
            print(output)
            
    return output
    
def _build(conf, input_channels):
    '''
    input: conf and feature extractor
    '''

    # height, slope, aspect, curvature
    output_channels=conf['data_output_channels']
    patch_size=conf['data_patch_size']
    
    # encoder
    input_x  = keras.layers.Input(shape=(patch_size, patch_size, input_channels))
    
    kernel_init=keras.initializers.glorot_normal()
    bias_init=keras.initializers.Constant(value=0.0)

    filters=conf['net_filter']
    kernel_size = conf['net_kernel']
    strides=conf['net_stride']

    print('==============encoder================')
    encoder, conv_list,_=encoder(input_x,filters, kernel_size, strides, kernel_init, bias_init)
    
    # decoder
    filters=list(reversed(filters))
    kernel_size=list(reversed(kernel_size))
    strides=list(reversed(strides))
    
    if conf['net_u_connections']:
        conv_list = list(reversed(conv_list))[1:]
        conv_list.append(None)
        print(conv_list)
    else:
        conv_list=None

    print('==============decoder================')
    decoder = decoder(encoder, filters,kernel_size,strides,1,kernel_init, bias_init, conv_list)

    # final output
    k=3
    if 'net_output_kernel' in conf.keys():
        if conf['net_output_kernel']:
            k=conf['net_output_kernel']
            
    # final output
    output=keras.layers.Conv2D(output_channels, [k,k], padding='same',kernel_initializer=kernel_init, bias_initializer=bias_init)(decoder)

    print(output)
    
    # added in 2020-12-03
    output=_add_mask_in_output(conf,input_x,output)
    
    # model compile
    model = keras.models.Model(input_x, output)
    if conf['train_optimizer'].lower()=='adam':
        optimizer=keras.optimizers.Adam
    elif conf['train_optimizer'].lower()=='rmsprop':
        optimizer=keras.optimizers.RMSprop
    else:
        optimizer=keras.optimizers.SGD
    
    model.compile(keras.optimizers.Adam(lr=conf['train_learning_rate']), loss=loss_func(conf))
    # model.summary()
    
    return model

def _build_segnet(conf, input_channels):
    '''
    input: conf and feature extractor
    '''
    
    # height, slope, aspect, curvature
    output_channels=conf['data_output_channels']
    patch_size=conf['data_patch_size']
    
    # encoder
    input_x  = keras.layers.Input(shape=(patch_size, patch_size, input_channels))
    
    kernel_init=keras.initializers.glorot_normal()
    bias_init=keras.initializers.Constant(value=0.0)

    filters=conf['net_filter']
    kernel_size = conf['net_kernel']
    strides=conf['net_stride']
    pool_pos=conf['net_pool_pos']
    
    activation_for_all=True
    # optional activation layer specification
    if 'net_activation_for_all' in conf.keys():
        activation_for_all=conf['net_activation_for_all']
    
    only_one_conv=False
    # optional specification of num. of conv layer after the up_sampling layer
    if 'net_fewest_conv_decoder' in conf.keys():
        only_one_conv=conf['net_fewest_conv_decoder']
    
    print('==============encoder================')
    encoder, layer_list,_=encoder_segnet(input_x,filters, kernel_size, strides, pool_pos, kernel_init, bias_init, activation_for_all)

    if not conf['net_u_connections']:
        layer_list=None
    print('==============decoder================')
    # optional kernel size override
    if 'net_kernel_size_decoder' in conf:
        k=conf['net_kernel_size_decoder']
        if isinstance(k, int):
            k=[k,k]
        kernel_size = [k for _ in kernel_size]
    # optional upsampling type specification
    convt=True
    if 'net_upsampling_type_decoder' in conf:
        if conf['net_upsampling_type_decoder']=='upsampling':
            convt=False
    # decoder
    decoder = decoder_segnet(encoder, filters,kernel_size,strides,pool_pos,kernel_init, bias_init, layer_list, convt, activation_for_all, only_one_conv)

    k=3
    if 'net_output_kernel' in conf.keys():
        if conf['net_output_kernel']:
            k=conf['net_output_kernel']
            
    # final output
    output=keras.layers.Conv2D(output_channels, [k,k], padding='same',kernel_initializer=kernel_init, bias_initializer=bias_init)(decoder)

    print(output)
    
    # added in 2020-12-03
    output=_add_mask_in_output(conf,input_x,output)
    
    if 'data_output_size_offset' in conf.keys():
        if conf['data_output_size_offset']:
            output=keras.layers.Cropping2D(conf['data_output_size_offset'])(output)
            print(output)
    
    # model compile
    model = keras.models.Model(input_x, output)
    if conf['train_optimizer'].lower()=='adam':
        optimizer=keras.optimizers.Adam
    elif conf['train_optimizer'].lower()=='rmsprop':
        optimizer=keras.optimizers.RMSprop
    else:
        optimizer=keras.optimizers.SGD
    
    model.compile(keras.optimizers.Adam(lr=conf['train_learning_rate']), loss=loss_func(conf))
    # model.summary()
    
    return model

def _build_segnet_no_shape(conf, input_channels):
    '''
    input: conf and feature extractor
    '''
    
    # height, slope, aspect, curvature
    output_channels=conf['data_output_channels']
    # patch_size=conf['data_patch_size']
    
    # encoder
    input_x  = keras.layers.Input(shape=(None, None, input_channels))
    
    kernel_init=keras.initializers.glorot_normal()
    bias_init=keras.initializers.Constant(value=0.0)

    filters=conf['net_filter']
    kernel_size = conf['net_kernel']
    strides=conf['net_stride']
    pool_pos=conf['net_pool_pos']
    
    print('==============encoder================')
    encoder, layer_list,_=encoder_segnet(input_x,filters, kernel_size, strides, pool_pos, kernel_init, bias_init)

    if not conf['net_u_connections']:
        layer_list=None
    print('==============decoder================')
    decoder = decoder_segnet(encoder, filters,kernel_size,strides,pool_pos,kernel_init, bias_init, layer_list)

    k=3
    if 'net_output_kernel' in conf.keys():
        if conf['net_output_kernel']:
            k=conf['net_output_kernel']
            
    # final output
    output=keras.layers.Conv2D(output_channels, [k,k], padding='same',kernel_initializer=kernel_init, bias_initializer=bias_init)(decoder)

    print(output)
    
    # added in 2020-12-03
    output=_add_mask_in_output(conf,input_x,output)
    
    # model compile
    model = keras.models.Model(input_x, output)
    if conf['train_optimizer'].lower()=='adam':
        optimizer=keras.optimizers.Adam
    elif conf['train_optimizer'].lower()=='rmsprop':
        optimizer=keras.optimizers.RMSprop
    else:
        optimizer=keras.optimizers.SGD
    
    model.compile(keras.optimizers.Adam(lr=conf['train_learning_rate']), loss=loss_func(conf))
    # model.summary()
    
    return model