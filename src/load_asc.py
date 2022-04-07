import gzip
import pandas as pd
import numpy as np

def read_asc(file):
    array = None
    if ".gz" in file:
        with gzip.open(file,"rb") as zipfile:
            array = pd.read_csv(zipfile,header=None,skiprows=6,delimiter=" ",skipinitialspace =True,dtype=np.float32).values
    else:
        array = pd.read_csv(file,header=None,skiprows=6,delimiter=" ",skipinitialspace =True,dtype=np.float32).values
    
    # some asc file contains empty columns, delete them
    if np.any(np.isnan(array[:,-1])):
        array = array[:,:-1]

    return array

# export asc file for analysis (Joao)

import io

def to_asc(array, cellsize=1, xllcorner=0, yllcorner=0, nodata=-9999.0, filename=None):
    str_buf=io.StringIO()
    np.savetxt(str_buf,array,delimiter=' ',fmt='%.8f')
    
    txt="\n".join(['ncols        %d'%array.shape[1],
                   'nrows        %d'%array.shape[0],
                   'xllcorner    %.8f'%xllcorner,
                   'yllcorner    %.8f'%yllcorner,
                   'cellsize     %.8f'%cellsize,
                   'NODATA_value  %.1f'%nodata,
                   str_buf.getvalue()
                  ])

    if filename is None:
        return txt
    else:
        with open(filename,'w') as f:
            f.write(txt)
            
def to_asc_partial(array,pos,size,cellsize=1,xllcorner=0,yllcorner=0,nodata=-9999.0,filename=None):
    hs=size//2
    row,col=pos
    return to_asc(array[row-hs:row+hs,col-hs:col+hs],
           xllcorner=xllcorner+(pos[0]-hs)*cellsize,
           yllcorner=yllcorner+(array.shape[1]-1-pos[1]-hs)*cellsize,
           cellsize=cellsize,
           nodata=nodata,
           filename=filename
          )
    
def read_asc_adv(filename):
    if filename.lower().endswith('.gz'):
        with gzip.open(filename) as f:
            array=pd.read_csv(f,header=None,skiprows=6,delimiter=" ",skipinitialspace =True,dtype=np.float32).values
    else:
        array=pd.read_csv(filename,header=None,skiprows=6,delimiter=" ",skipinitialspace =True,dtype=np.float32).values
    
    if np.any(np.isnan(array[:,-1])):
        array = array[:,:-1]
    
    if filename.lower().endswith('.gz'):
        with gzip.open(filename) as f:
            lines=f.readlines(1024)
        lines=[l.decode().strip() for l in lines]
    else:
        with open(filename) as f:
            lines=f.readlines(1024)
        lines=[l.strip() for l in lines]
        
    ncols=int(lines[0].split(' ')[-1])
    nrows=int(lines[1].split(' ')[-1])
    xllcorner=float(lines[2].split(' ')[-1])
    yllcorner=float(lines[3].split(' ')[-1])
    cellsize=float(lines[4].split(' ')[-1])
    nodata=float(lines[5].split(' ')[-1])
    
#     print(array.shape,nrows,ncols)
    
    return array, {'ncols':ncols,'nrows':nrows,'xllcorner':xllcorner,'yllcorner':yllcorner,'cellsize':cellsize,'nodata':nodata}