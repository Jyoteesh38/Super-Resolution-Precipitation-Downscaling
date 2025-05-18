import os
import random
os.environ["TF_DISABLE_NVTX_RANGES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["NCCL_DEBUG"] = "WARN"
import time
import tensorflow as tf
import numpy as np
import sys
import xarray as xr
import horovod.tensorflow.keras as hvd
import time
import socket
import math
import cupy as cp

from skimage.metrics import structural_similarity as ssim_ski
#------
hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
if hvd.local_rank() == 0:
    print("Socket and len gpus = ",socket.gethostname(), len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')   
#---------------------
cs = xr.open_mfdataset('pr_surf.ccam_12.5km.20*.nc', combine='by_coords', parallel=True)
rain = cs.pr.sel(time=slice('2012-09-16','2020-12-31'))
#-------------------------------------------------------------------------
#Orography-----------------------------------------------------------------
elev_file = xr.open_dataset('oro.nc')
elev = elev_file.oro.data
elev_norm = (elev/elev.max())
#-------------------------------------------------------------------------
Total_images = len(rain[:,0,0])
if hvd.rank() == 0:
    #nworkers = int(hvd.size())
    istart = int(hvd.rank()*Total_images/hvd.size())
    istop  = int((hvd.rank()+1)*Total_images/hvd.size())
else:
    istart = int(hvd.rank()*Total_images/hvd.size())
    istop  = int((hvd.rank()+1)*Total_images/hvd.size())
if istop >= Total_images:
    istop = Total_images - 1
print ( '*** rank = ', hvd.rank(),' istart = ', istart, ' istop = ', istop)
#-------
#---------------------------
x_tr = 100*21.9587176109*np.expand_dims(rain[istart:istop,:,:].data, axis=3)
shrink = 8
x = tf.keras.layers.AveragePooling2D(
          pool_size=(shrink, shrink), strides=None, padding='same', data_format=None)(x_tr)
#Matching oro data length with the train and test lengths
#orography data
pq = np.expand_dims(elev_norm, axis=2)
#---------------------
folder = 'epoch_runs/'
DL_models = ['DEEPSD', 'DEEPSD_100','SRDCNN', 'SRDCNN_ORO', 'SRDCNN_STEP_ORO']
def model_predict(m_num):
   #---------------
    if m_num > 0:
        p_oro = 100*np.repeat(pq[np.newaxis, :, :, :], x.shape[0], axis=0)
    else:
        p_oro = np.repeat(pq[np.newaxis, :, :, :], x.shape[0], axis=0)
   #---------------
    if m_num < 2:
        part_1 = tf.keras.models.load_model(folder+DL_models[m_num]+'_p1.h5', compile=False)
        part_2 = tf.keras.models.load_model(folder+DL_models[m_num]+'_p2.h5', compile=False)
        part_3 = tf.keras.models.load_model(folder+DL_models[m_num]+'_p3.h5', compile=False)
        rain_1 = part_1.predict([x, p_oro])
        rain_2 = part_2.predict([rain_1, p_oro])
        del rain_1
        model_rain = part_3.predict([rain_2, p_oro])
        del rain_2
    elif m_num == 2:
        model = tf.keras.models.load_model(folder+DL_models[m_num]+'.h5', compile=False)
    if hvd.rank() == 0:
        model.summary()
        model_rain = model.predict(x)
    else:
        model = tf.keras.models.load_model(folder+DL_models[m_num]+'.h5', compile=False)
    if hvd.rank() == 0:
        model.summary()
        model_rain = model.predict([x, p_oro])
   #-------------------- 
    del p_oro
    model_rain = 1.63944*(model_rain.clip(min=0))
    print('*** rank = ', hvd.rank(),'predict shape:',model_rain.shape)
   #-------------------------
    def fft_mean(rx,ry):
        cp_fft = cp.zeros_like(cp.asarray(rx[0,:,:]))
        cp_rain = cp.asarray(rx)
        iend = int(len(cp_rain[:,0,0]))
        print('*** rank = ', hvd.rank(),'loop length=', iend)
        for i in range(0, iend):
            fft = cp.square(cp.absolute(cp.fft.fftshift(cp.fft.fft2(cp_rain[i,:,:]))))
            cp_fft += fft
        del cp_rain
        y = cp_fft/iend
        del cp_fft
        z = xr.zeros_like(ry)
        z[:,:] = cp.asnumpy(y)
        del y
        return z
   #-------------------------
    def clim_mean(rx, ry):
        cp_rain = cp.asarray(rx)
        precip = xr.zeros_like(ry)	
        precip[:,:] = cp.asnumpy(cp.mean(cp_rain, axis=0))
        del cp_rain
        return precip
   #-------------------------
    def mse(rx, ry, x):
        mse = xr.zeros_like(ry)
        mse[:,:] = tf.reduce_mean((tf.math.square(rx - (1.63944*x[:,:,:,0]))), axis=0)
        return mse
   #-------------------------
    def psnr_ssim(rx, x):
        x = 1.63944*x
        iend = int(len(rx[:,0,0,0]))
        psnr_sum = 0
        ssim_sum = 0
        for i in range(0, iend):
            max_val = tf.reduce_max(x[i,:,:,:])
            psnr = tf.image.psnr(x[i,:,:,:], rx[i,:,:,:], max_val=max_val)
            psnr_sum += psnr
            ssim = ssim_ski(x[i,:,:,0],rx[i,:,:,0], data_range=max_val)
            #print('*** rank = ', hvd.rank(),'predict shape:',ssim.shape)
            ssim_sum += ssim
        return psnr_sum/iend, ssim_sum/iend
   #-------------------------
    mean_fft = fft_mean(model_rain[:,:,:,0],rain[0,:,:])
    climmean = clim_mean(model_rain[:,:,:,0],rain[0,:,:])
    rms = mse(model_rain[:,:,:,0],rain[0,:,:],x_tr)
   #-------------------------
    pr_model = xr.zeros_like(rain[istart:istop,:,:])
    pr_model[:,:,:] = model_rain[:,:,:,0]
   #-------------------------
    psr, sim = psnr_ssim(model_rain, x_tr)
    print('*** rank = ',hvd.rank(),DL_models[m_num]+'_psnr = ',psr,DL_models[m_num]+'_ssim = ',sim)
    del model_rain
   #-------------------------
    pr_model.to_netcdf(folder+'Data/'+DL_models[m_num]+'_PR_model_{}.nc'.format(hvd.rank()))
    del pr_model
    mean_fft.to_netcdf(folder+'Data/'+DL_models[m_num]+'_fft_{}.nc'.format(hvd.rank()))
    del mean_fft
    climmean.to_netcdf(folder+'Data/'+DL_models[m_num]+'_clim_{}.nc'.format(hvd.rank()))
    del climmean
    rms.to_netcdf(folder+'Data/'+DL_models[m_num]+'_MSE_{}.nc'.format(hvd.rank()))
    del rms
   #-------------------------
    print('*** rank = ', hvd.rank(),'prediction completed')
    return psr, sim
#---------------
#0:DEEPSD; 1:DEEPSD_100
#2:SRDCNN; 3:SRDCNN_ORO
#4: SRDCNN_STEP_ORO
for i in range(5):
    model_predict(i)
#---------------
