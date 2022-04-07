import keras
import numpy as np
import cv2

class PatchFromFile:
    '''
    access patch from pre-processed npy files
    
    be careful with the fortran_order and little_endian options
    
    fortran_order: the bytes of the nd-array are stored in a reversed order,
                   e.g., shape(a,b,c) cannot be reshaped using reshape(a,b,c),
                   but have to use transpose(reshape(c,b,a))
    
    little_endian (will be included soon): see https://en.wikipedia.org/wiki/Endianness
    '''
    def __init__ (self, files, shape, dtype, fortran_order=False, batch_size=8, repeat = None, parallel=2, shuffle=None, prefetch=1):
        data_loader = load_dataset_npy(files,shape,dtype,fortran_order=fortran_order, num_parallel_calls=parallel)
        if shuffle is not None:
            data_loader = data_loader.shuffle(shuffle)
        
        self.patch_num = len(files)
        self.current = 0 # counter for end_of_epoch(), an optional function that does not effect the patch generation
        self.batch_size = batch_size
        
        data_loader = data_loader.batch(batch_size).prefetch(prefetch).repeat(repeat)
        self.data_loader = data_loader
        self.get_next_op = data_loader.make_one_shot_iterator().get_next()
    
    def set_session(self, sess):
        self.sess = sess
    
    def next_batch(self):
        arr = self.sess.run(self.get_next_op)
        self.current += len(arr)
        return arr
    
    def end_of_epoch(self):
        '''
        indicate if current epoch has completed, this is an indicator function to simplify the
        training code, it does not effect the patch generation
        
        call end_of_epoch before next_batch to get correct result
        
        for example:
        
        while not end_of_epoch():
            next_batch()
            ...
        '''
        if self.current >= self.patch_num:
            self.current = self.current % self.patch_num
            return True
        else:
            return False

def _generate_patch_location(dem_array, patch_size, key_channel, key_value, patch_num, ratio):
    height, width, _ = dem_array.shape
    
    if patch_num is None:
        patch_num = ratio * (height * width) // (patch_size**2)

    patches = []
    i=0
    while i < patch_num:
        w = np.random.randint(width - patch_size)
        h = np.random.randint(height - patch_size)
        
        if key_channel is not None:
            if np.all(dem_array[h:h+patch_size,w:w+patch_size,key_channel]==key_value):
                # not any 1 within the area?
                continue
        
        i+=1
        patches.append([h,w])
    return np.array(patches, dtype=np.uint32)

def _prob_inverse_x(a=10):
    '''
    prob(x)=1-1/(x.mean()+1)^a
    with larger a, smaller x tends to have higher probablity
    
    with a=10
    x.mean() | prob
    0.01 | 0.095
    0.1 | 0.61
    0.2 | 0.84
    0.3 | 0.93
    0.4 | 0.97
    
    with a=5
    x.mean() | prob
    0.01 | 0.049
    0.1 | 0.38
    0.2 | 0.60
    0.3 | 0.73
    0.4 | 0.81
    '''
    return lambda x:1-1/((x.mean()+1)**a)

def _generate_patch_location_weighted(dem_array, patch_size, key_channel, key_value, patch_num, ratio, weight_func, weight_channel,max_trials=100):
    '''
    key_channel: channel(s) that detect the no-data pixels
    key_value: values being filled for no-data pixels
    
    weight_func: a function returns the prob of a given patch being accepted
    weight_channel: on which channel the weight_func is applied
    '''
    height, width, _ = dem_array.shape
    
    if patch_num is None:
        patch_num = max(1,int(ratio * (height * width) / (patch_size**2)))
        print(patch_num,'patches for input',height,'x',width,'with ratio',ratio,end=' ')

    patches = []
    i=0
    t=0
    
    while i < patch_num:
        w = np.random.randint(width - patch_size)
        h = np.random.randint(height - patch_size)
        
        if key_channel is not None:
            if np.all(dem_array[h:h+patch_size,w:w+patch_size,key_channel]==key_value):
                # not any 1 within the area?
                continue
        
        w=weight_func(dem_array[h:h+patch_size,w:w+patch_size,weight_channel])
        if np.random.random() > w:
            # fail to pass the weight check
            if t<max_trials:
                t+=1
                continue
                
        # reset t after a successful sampling
        t=0

        i+=1
        patches.append([h,w])
    print(w,' ',end='\r') # for debug
    return np.array(patches, dtype=np.uint32)

def _generate_patch_location_wheel_weighted(dem_array, patch_size, key_channel, key_value, patch_num, ratio, base_prob=1e-4, pow_prob=2):
    height, width, _ = dem_array.shape

    prob_map=np.zeros((height,width))
    mask=np.logical_not(np.all(dem_array[:,:,key_channel]==key_value,axis=-1))
    prob_map[mask]=np.power(dem_array[...,-2][mask],pow_prob)
    prob_map[mask]+=base_prob
    
    prob_map=prob_map.flatten()
    total=prob_map.sum()
    
    if patch_num is None:
        patch_num = max(1,int(ratio * (height * width) / (patch_size**2)))
        print('prob min',prob_map.min(),'max',prob_map.max(),'total',total,end='\r')

    patches = []

    i=0
    while i < patch_num:
        k=np.random.rand()*total
        
        # wheel selection
        selected=0
        accu=0
        for idx in range(0,prob_map.shape[0]):
            accu+=prob_map[idx]
            if accu>=k:
                selected=idx
                break
        
        h=selected//width
        w=selected%width
        
        h-=patch_size//2
        w-=patch_size//2
        
        h=max(h,0)
        w=max(w,0)
        
        h=min(h,height-1-patch_size)
        w=min(w,width-1-patch_size)
        
        i+=1
        patches.append([h,w])
    # print(patches)
    return np.array(patches, dtype=np.uint32)

def uniform_patch_location_generator(patch_size, key_channel, key_value, num, ratio):
    return lambda x: _generate_patch_location(x, patch_size, key_channel, key_value, num, ratio)

def weighted_patch_location_generator(patch_size, key_channel, key_value,num, ratio, weight_func, weight_channel):
    return lambda x: _generate_patch_location_weighted(x, patch_size, key_channel, key_value, num, ratio, weight_func, weight_channel)

def wheel_weighted_patch_location_generator(patch_size, key_channel, key_value, num, ratio):
    return lambda x: _generate_patch_location_wheel_weighted(x, patch_size, key_channel, key_value, num, ratio)

class PatchFromImage(keras.utils.Sequence):
    '''
    
    a keras implementation of data-generation
    
    see https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly for details
    
    generate patch from pre-loaded multi-channel images
    
    note: the input image must be rank 3 image, even there is only one channel!
    '''
    def __init__(self, image_list, patch_size, output_channels=1, batch_size=8, seed=None, shuffle=True, augmentation=False, key_channel=Ellipsis, key_value=0, num=None, ratio=5, patch_loc_generator=None, output_size_offset=0, regen_patch_every_epoch=False):
        '''
        update in 2020-06-12
        
        if patch_loc_generator is None, then a default patch generator (uniform random) is created using patch_size, key_channel, key_value, num, ratio. otherwise, the assigned patch_loc_generator will be used and key_channel, key_value, num, ratio will have no effect

        output_size_offset: if in the case output patch has different size of (smaller or larger than) input patch, specify this value
        by default offset towards the inside of the patch (output_size_offset>0).
        
        the patch_loc_generator must have the same patch_size as given in this function, unless output_size_offset != 0. In this case, the patch_loc_generator gives the offseted location [output_size_offset:patch_size-output_size_offset]
        
        the input images must at least host the size of [output_size_offset:patch_size-output_size_offset]
        
        update in 2020-06-19
        
        not-a-bug fix: data augmentation incluse t variable, giving 8 (instead of 4) patch variants including the original one
        '''
        # set seed
        if seed is not None:
            np.random.seed(seed)
            
        self.image_list = image_list
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.output_channels = output_channels
        
        self.output_size_offset=output_size_offset
        # channel size
        self.image_channels=image_list[0].shape[-1]
        
        if patch_loc_generator is None:
            patch_loc_generator=uniform_patch_location_generator(patch_size-2*output_size_offset, key_channel, key_value, num, ratio)
        self.patch_loc_generator=patch_loc_generator
        
        self.augmentation=augmentation
        self.regen_patch_every_epoch=regen_patch_every_epoch
        
        # generate patches
        self._gen_patches()
        
        print (self.patch_num, "patches in total")
        # initiate the indice
        self.do_shuffle()

    def _gen_patches(self):
        patch_loc_generator=self.patch_loc_generator
        image_list=self.image_list
        augmentation=self.augmentation
        
        # patch location for each nd_image
        # locs is a 2 level list, the 1st level indicates which image, the 2nd level indicates which patch
#         locs = [generate_patch_location(arr, patch_size, key_channel, key_value, num, ratio) for arr in image_list]
        locs = [patch_loc_generator(arr) for arr in image_list]
    
        # insert the index of the corresponding image to the end of the patch location
        # locs is a 2 level list, the 2nd level has length of 3
        locs = [np.concatenate([locs[i], i * np.ones((len(locs[i]),1),dtype=np.uint32)], axis=-1) for i in range(len(locs))]
        
        # flatten the list to one level only
        self.patch_locations = np.concatenate(locs,axis=0)

        # data augmentation, insert x, y flip plus transpose at the end of the patch location (= 4 rotations with mirror, or 8 results)
        if augmentation:
            self.patch_locations=np.array([[h,w,i,y,x,t] for x in [-1,1] for y in [-1,1] for t in [0,1] for h,w,i in self.patch_locations])
            
        self.patch_num = len(self.patch_locations)
        
    def __len__(self):
        return self.patch_num // self.batch_size
    
    def on_epoch_end(self):
        if self.regen_patch_every_epoch:
            print('regen-patches')
            self._gen_patches()
        self.do_shuffle()
    
    def do_shuffle(self):
        '''
        shuffle the patch locations in axis 0
        '''
        if self.shuffle:
            np.random.shuffle(self.patch_locations)
        self.start = 0
    
    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index+1) * self.batch_size
        locs = self.patch_locations[start:end]
        
        if self.output_size_offset==0:
            # input and output patches has the same size
            if self.augmentation:
                # take 4 types of flip and transpose into consideration
                arr = np.array([self.image_list[i][h:h+self.patch_size, w:w+self.patch_size][::y,::x] if t==0 else np.transpose(self.image_list[i][h:h+self.patch_size, w:w+self.patch_size][::y,::x],[1,0,2]) for h,w,i,y,x,t in locs])
            else:
                arr = np.array([self.image_list[i][h:h+self.patch_size, w:w+self.patch_size] for h,w,i in locs])

            return arr[...,:-self.output_channels], arr[...,-self.output_channels:]
        elif self.output_size_offset>0:
            # output size is smaller
            batch_size=self.batch_size
            patch_size=self.patch_size
            offset=self.output_size_offset
            
            image_channels=self.image_channels
            output_size=patch_size-offset*2
            
            # a buffer array
            arr_in=np.zeros((batch_size,patch_size,patch_size,image_channels),dtype=np.float32)
            
            if self.augmentation:
                count=0
                for h,w,i,y,x,t in locs:
                    img_shape=self.image_list[i].shape[:2]
                    
                    # the output patch is sampled based on patch_size-offset*2, in case the input patch out of bound, a check is needed
                    h_start=max(0,h-offset)
                    h_start_local=h_start-(h-offset)
                    
                    h_end=min(img_shape[0],h+patch_size-offset)
                    h_end_local=h_end-h_start+h_start_local
                    
                    w_start=max(0,w-offset)
                    w_start_local=w_start-(w-offset)
                    
                    w_end=min(img_shape[1],w+patch_size-offset)
                    w_end_local=w_end-w_start+w_start_local
                    
                    # copy paste data within the checked bound, data outside the bound are left with 0s
                    arr_in[count,h_start_local:h_end_local,w_start_local:w_end_local]=self.image_list[i][h_start:h_end, w_start:w_end]
                    arr_in[count]=arr_in[count,::y,::x] # flip
                    
                    if t>0:
                        arr_in[count]=np.transpose(arr_in[count],[1,0,2])
                        
                    count+=1
                    
                arr_out = np.array([self.image_list[i][h:h+output_size, w:w+output_size][::y,::x] if t==0 else np.transpose(self.image_list[i][h:h+output_size, w:w+output_size][::y,::x],[1,0,2]) for h,w,i,y,x,t in locs])
            else:
                count=0
                for h,w,i in locs:
                    img_shape=self.image_list[i].shape[:2]
                    # unoffset the h and w so that it reaches maximum patch size x patch size
                    h_start=max(0,h-offset)
                    h_start_local=h_start-(h-offset) #the location of unoffset array in patch size x patch size
                    
                    h_end=min(img_shape[0],h+patch_size-offset)
                    h_end_local=h_end-h_start+h_start_local
                    
                    w_start=max(0,w-offset)
                    w_start_local=w_start-(w-offset)
                    
                    w_end=min(img_shape[1],w+patch_size-offset)
                    w_end_local=w_end-w_start+w_start_local
                    
                    arr_in[count,h_start_local:h_end_local,w_start_local:w_end_local]=self.image_list[i][h_start:h_end, w_start:w_end]
                    count+=1
                    
                arr_out = np.array([self.image_list[i][h:h+output_size, w:w+output_size] for h,w,i in locs])
                
            # return the input and output arrays
            return arr_in[...,:-self.output_channels], arr_out[...,-self.output_channels:]
        else:
            raise Exception('not implemented yet')
        
class PatchFromImageWithRotation(keras.utils.Sequence):
    '''
    
    a keras implementation of data-generation
    
    see https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly for details
    
    generate patch from pre-loaded multi-channel images
    
    note: the input image must be rank 3 image, even there is only one channel!
    '''
    def __init__(self, image_list, patch_size, output_channels=1, batch_size=8, seed=None, shuffle=True, augmentation=False, rot_step=10, flip_x=[-1,1],flip_y=[-1,1],key_channel=Ellipsis, key_value=0, num=None, ratio=5,patch_loc_generator=None,regen_patch_every_epoch=False):
        # set seed
        if seed is not None:
            np.random.seed(seed)
            
        self.image_list = image_list
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.output_channels = output_channels
        self.debug=False
        
        if patch_loc_generator is None:
            patch_loc_generator=uniform_patch_location_generator(patch_size, key_channel, key_value, num, ratio)
        
        self.patch_loc_generator=patch_loc_generator
        self.regen_patch_every_epoch=regen_patch_every_epoch
        self.augmentation=augmentation
        
        self._gen_patches()
        
        print (self.patch_num, "patches in total")
        # initiate the indice
        self.do_shuffle()

    def _gen_patches(self):
        patch_loc_generator=self.patch_loc_generator
        image_list=self.image_list
        augmentation=self.augmentation
        # patch location for each nd_image
        # locs is a 2 level list, the 1st level indicates which image, the 2nd level indicates which patch
        locs = [patch_loc_generator(arr) for arr in image_list]
        
        # insert the index of the corresponding image to the end of the patch location
        # locs is a 2 level list, the 2nd level has length of 3
        locs = [np.concatenate([locs[i], i * np.ones((len(locs[i]),1),dtype=np.uint32)], axis=-1) for i in range(len(locs))]
        
        # flatten the list to one level only
        self.patch_locations = np.concatenate(locs,axis=0)
        
        # data augmentation, insert x and y flip direction at the end of the patch location
        if augmentation:
            self.patch_locations=np.array([[h,w,i,y,x,r] for x in flip_x for y in flip_y for r in range(0,360,rot_step) for h,w,i in self.patch_locations])
            
        self.patch_num = len(self.patch_locations)
        
    def __len__(self):
        return self.patch_num // self.batch_size
    
    def on_epoch_end(self):
        if self.regen_patch_every_epoch:
            print('regen-patches')
            self._gen_patches()
        self.do_shuffle()
    
    def do_shuffle(self):
        '''
        shuffle the patch locations in axis 0
        '''
        if self.shuffle:
            np.random.shuffle(self.patch_locations)
        self.start = 0
    
    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index+1) * self.batch_size
        locs = self.patch_locations[start:end]
        patch_size=self.patch_size
        
        if self.augmentation:
            arr=[]
            # take 4 types of flip into consideration
            for h,w,i,y,x,r in locs:
                shape = self.image_list[i].shape
                pad=int(np.ceil(patch_size * ((np.abs(np.sin(r/180*3.1415926))+np.abs(np.cos(r/180*3.1415926)))-1) / 2))
                #if self.debug:
                    #print(h,w,i,y,x,r,pad)
                if h-pad >= 0 and w-pad >= 0 and h+patch_size+pad < shape[0] and w+patch_size+pad < shape[1]:
                    # can rotate, get the padded image
                    img = self.image_list[i][h-pad:h+patch_size+pad, w-pad:w+patch_size+pad][::y,::x]
                    rows,cols = img.shape[:2]
                    # cols-1 and rows-1 are the coordinate limits.
                    M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),r,1)
                    # rotation
                    rot_img = cv2.warpAffine(img,M,(cols,rows))
                    # unpad
                    rot_img = rot_img[pad:pad+patch_size, pad:pad+patch_size]
                    arr.append(rot_img)
                    if self.debug:
                        print(r)
                else:
                    # cannot rotate, skip rotation
                    arr.append(self.image_list[i][h:h+patch_size, w:w+patch_size][::y,::x])
            arr = np.array(arr)
        else:
            arr = np.array([self.image_list[i][h:h+patch_size, w:w+patch_size] for h,w,i in locs])
            
        return arr[...,:-self.output_channels], arr[...,-self.output_channels:]