# some validation functions that can be shared between different experiments

import numpy as np
import os
import json

def predict_catchment_patch(model,
                            nd_array,
                            output_channels,
                            patch_size,
                            batch_size=2,
                            grid_size=None,
                            random_patch_num=None,
                            return_patch_num=False):
    
    height, width = nd_array.shape[:2]
    print(height,width)
    
    if grid_size is None:
        grid_size=patch_size//2
        
    # necessary data for prediction
    y_mean = np.zeros((height,width,output_channels), dtype=np.float32)
    y_n = np.zeros((height,width,output_channels), dtype=np.float32)
    
    # generate patches
    # patch_size+2 is the size for small catchment after padding
    h_pos=[h for h in range(0,height-patch_size,grid_size)]
    if h_pos[-1]<height-patch_size-1 and height>patch_size+2:
        h_pos.append(height-patch_size-1)
        
    w_pos=[w for w in range(0,width-patch_size,grid_size)]
    if w_pos[-1]<width-patch_size-1 and width>patch_size+2:
        w_pos.append(width-patch_size-1)

    patches_grid = np.array([[h,w] for h in h_pos for w in w_pos])
    
    # generate random patch locations
    if random_patch_num is None:
        random_patch_num=len(patches_grid)
        
    if random_patch_num > 0 and len(patches_grid)>1:
        patches_rand = np.transpose([np.random.randint(0,height-patch_size,random_patch_num),
                        np.random.randint(0,width-patch_size,random_patch_num)])

        # merge grid-based and random locations
        patches=np.concatenate([patches_grid,patches_rand],axis=0)
    else:
        patches=patches_grid
        
    # select those that contains catchment areas
    patches =  np.array([[h, w] for h, w in patches if np.any(nd_array[h:h+patch_size,w:w+patch_size,0]!=0)])
    num_patches = len(patches)

    for i in range(0, num_patches, batch_size):
        start = i
        end = min(i+batch_size, num_patches)
        
        cur_batch = patches[start:end]
        x_batch = np.array([nd_array[h:h+patch_size,w:w+patch_size] for h,w in cur_batch])
        
        y_batch = model.predict(x_batch[...,:-output_channels]) # remove ground truth from input
        y_batch = y_batch[...,:output_channels] # remove extra channels (e.g., concatenated mask channel) from output
        
        for j in range(len(cur_batch)):
            h,w = cur_batch[j]
            y_mean[h:h+patch_size,w:w+patch_size] += y_batch[j,...]
            y_n[h:h+patch_size,w:w+patch_size] += 1

    y_mean[y_n>0] /= y_n[y_n>0]
    
    if not return_patch_num:
        return y_mean
    else:
        return y_mean, num_patches

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        
def test_folder(conf,ensure=True):
    test_folder = os.path.join("tests",conf['type'], conf['name'])
    if ensure:
        ensure_folder(test_folder)
    return test_folder

def model_folder(conf,ensure=True):
    model_folder = os.path.join("trained_models",conf['type'], conf['name'])
    if ensure:
        ensure_folder(model_folder)
    return model_folder

# detect no-data pixels as they are filled with 0 in all image channels
MASK_ALL_ZERO=lambda x:np.logical_not(np.all(x==0,axis=-1))

# detect no-data pixels by mask channel, not implemented yet
MASK_CHANNEL_MASK=None

def validation_with_test_set(model,
                             conf,
                             batch_of_data,
                             matrix_max,
                             matrix_res,
                             masker=MASK_ALL_ZERO,
                             output_channel_name=None,
                             data_name='validation_set'
                            ):
    '''
    generate confution matrix, mae results from data batches (batch of training/test data)
    
    
    batch_of_data must have length property (len(batch_of_data)=integer)
    
    each batch_of_data must return batch_x, batch_y, in which batch_x is feed to model to get batch_y_pred
    '''
    target_folder=os.path.join(test_folder(conf),data_name)
    ensure_folder(target_folder)
    
    output_channel = conf['data_output_channels']
    
    if output_channel_name is None:
        output_channel_name=['channel %d'%i for i in range(output_channel)]
    else:
        if len(output_channel_name)<output_channel:
            raise Exception('must have at least %d elements in output_channel_name'%output_channel)
    
    matrix=[None]*output_channel
    error_accu=np.asarray([0.0]*output_channel)
    shape_accu=0
    
    _test = []
    _test2 = []
    
    length=len(batch_of_data)
    for i in range(length):
        batch_x, batch_y=batch_of_data[i]
        batch_y_p=model.predict(batch_x)[...,:output_channel] # added in 2020-12-03 [...,:output_channel]
        
        _error=np.abs(batch_y_p-batch_y)
        _error2=np.square(batch_y_p-batch_y)
        
        # for each element in the batch, conf matrix and 
        for j in range(batch_x.shape[0]):            
            mask=masker(batch_x[j]) if masker is not None else None
            
            _test.append(_error[j])
            _test2.append(_error2[j])
            
            if 'data_output_size_offset' in conf.keys():
                offset=conf['data_output_size_offset']
                if offset>0:
                    mask=mask[offset:-offset,offset:-offset]
            
            for k in range(output_channel):
                matrix[k] = accumulate_confusion_matrix(batch_y[j,...,k],
                                                        batch_y_p[j,...,k],
                                                        matrix_max[k]*matrix_res[k]+1, # plus one to enclude the ending value
                                                        res=matrix_res[k],
                                                        mask=mask,
                                                        current_matrix=matrix[k])
        
        error_accu+=np.sum(np.abs(batch_y_p-batch_y),axis=(0,1,2))
        shape_accu+=(batch_y.shape[0]*batch_y.shape[1]*batch_y.shape[2])
    
    _test=np.asarray(_test)
    _test2=np.asarray(_test2)
    print(_test.shape,_test2.shape)
    
    print("MAE",_test[...,0].mean(),_test[...,1].mean())
    print("MSE",_test2[...,0].mean(),_test2[...,1].mean())
    
    mae_channel=error_accu/shape_accu
    mae_avg=error_accu.sum()/(shape_accu*output_channel)
    
    mae_json={}
    # export the confusion matrix
    for i in range(output_channel):
        hist_confusion_matrix(matrix[i],matrix_max[i]*matrix_res[i]+1,matrix_res[i],
                              suptitle=conf['name']+'_'+output_channel_name[i]+'_matrix',
                              filename=os.path.join(target_folder, output_channel_name[i]+'_matrix.svg'))
        
        np.save(os.path.join(target_folder, output_channel_name[i]+'_matrix.npy'),matrix[i])
        
        mae_json[output_channel_name[i]]=float(mae_channel[i])
    mae_json['mae_avg']=float(mae_avg)
    
    # export the mae
    with open(os.path.join(target_folder, 'mae.json'),'w') as f:
        json.dump(mae_json,f)
    
    
from sklearn.metrics import confusion_matrix

def accumulate_confusion_matrix(truth,pred,classes_max,res=10,mask=None,current_matrix=None):
    # from float to integer
    int_truth=np.round(truth*res).astype(np.int32)
    int_pred=np.round(pred*res).astype(np.int32)
    # clip
    int_truth[int_truth>=classes_max]=classes_max-1
    int_pred[int_pred>=classes_max]=classes_max-1
    int_pred[int_pred<0]=0
    # mask
    if mask is not None:
        int_truth=int_truth[mask]
        int_pred=int_pred[mask]
    else:
        int_truth=int_truth.flatten()
        int_pred=int_pred.flatten()
    # the location of each pair in matrix
    # truth > row, pred > col, indice = row*col_size + col
    indice_in_matrix=int_truth*classes_max+int_pred
    # confusion matrix as bincount (2.5 times faster than sklearn's confusion matrix)
    matrix=np.bincount(indice_in_matrix,minlength=classes_max*classes_max).reshape(classes_max,classes_max)

    if current_matrix is None:
        current_matrix = matrix
    else:
        current_matrix += matrix
    return current_matrix

from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import Grid, ImageGrid

from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

TICK_FONT_SIZE=8
LABEL_FONT_SIZE=16
TITLE_FONT_SIZE=32

class nlcmap(object):
    def __init__(self, cmap, levels):
        self.cmap = cmap
        #self.N = cmap.N
        #self.monochrome = self.cmap.monochrome
        self.levels = np.asarray(levels, dtype='float64')
        #self._x = self.levels
        self.levmax = self.levels.max()
        self.levmin = self.levels.min()
        self.transformed_levels = np.linspace(self.levmin, self.levmax,
             len(self.levels))
        self.name="nlcmap"

    def __call__(self, xi, alpha=1.0, **kw):
        yi = np.interp(xi, self.levels, self.transformed_levels)
        return self.cmap((yi-self.levmin) / (self.levmax-self.levmin), alpha)

# code here https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_truth_pred_error(y_pred,y_truth,mask=None,enlargement=None,
                          size=(8,4),title=None,suptitle=None,
                          tickfontsize=TICK_FONT_SIZE,
                          labelfontsize=LABEL_FONT_SIZE,
                          titlefontsize=TITLE_FONT_SIZE,
#                           pred_truth_levels=None,
#                           error_levels=None,
                          show=False,filename=None):
    '''
    enlargement: [col, row, width, height]
    
    https://matplotlib.org/3.1.1/gallery/axes_grid1/demo_axes_grid.html
    
    wait to finish (cmap part)
    '''
    if mask is not None:
        y_pred=np.copy(y_pred)
        y_truth=np.copy(y_truth)
        
        y_pred[mask]=np.nan
        y_truth[mask]=np.nan

    error=y_pred-y_truth
    
    rows=y_pred.shape[-1]
    cols=3
    
    plt.close()
    fig=plt.figure(figsize=size)

    # (start x, start y, width, height)
    grid = ImageGrid(fig, (0,1,1,1) , # (num_catch+1,1,i+2),row-col specification
                     nrows_ncols=(rows,cols),
                     axes_pad=0.15,
                     share_all=False,
                     label_mode='L',
                     cbar_location="right",
                     cbar_mode="each",
                     cbar_size="5%",
                     cbar_pad=0.15,
                     )
    
    for i in range(rows):
        ax=grid[0+i*cols] # pred wd
        im = ax.imshow(y_pred[...,i],vmin=0,vmax=3)
        ax.tick_params(labelsize=tickfontsize)
        if enlargement is not None:
            rect = Rectangle((enlargement[0],enlargement[1]),enlargement[2],enlargement[3],
                             linewidth=3,edgecolor='r',facecolor='none')
            ax.add_patch(rect)

        ax=grid[1+i*cols] # truth wd
        im = ax.imshow(y_truth[...,i],vmin=0,vmax=3)
        ax.tick_params(labelsize=tickfontsize)
        if enlargement is not None:
            rect = Rectangle((enlargement[0],enlargement[1]),enlargement[2],enlargement[3],
                             linewidth=3,edgecolor='r',facecolor='none')
            ax.add_patch(rect)

        ax=grid[2+i*cols] # error wd
        im = ax.imshow(error[...,i],vmin=-2,vmax=2)
        ax.tick_params(labelsize=tickfontsize)
        if enlargement is not None:
            rect = Rectangle((enlargement[0],enlargement[1]),enlargement[2],enlargement[3],
                             linewidth=3,edgecolor='r',facecolor='none')
            ax.add_patch(rect)

    if filename is not None:
        plt.savefig(filename, dpi=300,pad_inches=0,bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def hist_confusion_matrix(matrix,classes_max,res,
                          figsize=(8,8),show_step=4,show_log=True,
                          unit_str=None,suptitle=None,
                          tickfontsize=TICK_FONT_SIZE,
                          labelfontsize=LABEL_FONT_SIZE,
                          titlefontsize=TITLE_FONT_SIZE,
                          show=False,filename=None):
    # reference plot:
    # https://matplotlib.org/3.1.1/gallery/axes_grid1/scatter_hist_locatable_axes.html#sphx-glr-gallery-axes-grid1-scatter-hist-locatable-axes-py
    classes = range(0, classes_max)
    labels=[str(i/res) for i in classes]

    # show x-y ticks for every show_step classes
    classes_to_show = [c for c in classes if c % show_step == 0]
    labels_to_show = [labels[c] for c in classes_to_show]

    plt.close()
    fig, axScatter = plt.subplots(figsize=figsize)
    # the main plot
    if show_log:
        im = axScatter.imshow(matrix,cmap='pink',norm=SymLogNorm(10))
    else:
        matrix=matrix+1 # to prevent divided by 0
        matrix_norm = matrix / matrix.sum(axis=1)
        im = axScatter.imshow(matrix_norm,cmap='pink')
        
    axScatter.scatter(range(len(classes)), range(len(classes)), color='white', s=10, marker='.') # diagonal
    
    # x-y limits
    axScatter.set_xlim([-0.5,len(classes)-0.5])
    axScatter.set_ylim([-0.5,len(classes)-0.5])

    # x-y labels
    axScatter.set_xlabel('pred' if unit_str is None else 'pred (' + unit_str + ')',fontsize=labelfontsize)
    axScatter.set_xticks(classes_to_show)
    axScatter.set_xticklabels(labels_to_show, fontdict ={'fontsize':tickfontsize})

    axScatter.set_ylabel('truth' if unit_str is None else 'truth (' + unit_str + ')',fontsize=labelfontsize)
    axScatter.set_yticks(classes_to_show)
    axScatter.set_yticklabels(labels_to_show,fontdict ={'fontsize':tickfontsize})
    axScatter.set_aspect(1)

    divider = make_axes_locatable(axScatter)
    axColorbar = divider.append_axes("right", 0.2, pad=0.1)

    plt.colorbar(im, cax=axColorbar)
    
    if suptitle is not None:
        plt.suptitle(suptitle,y=0.94,fontsize=titlefontsize)
    if filename is not None:
        plt.savefig(filename, dpi=300, pad_inches=0,bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()