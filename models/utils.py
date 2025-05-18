import xarray as xr
import numpy as np
import tensorflow as tf

def normalize_orography(path):
    elev_data = xr.open_dataset(path)
    elev = elev_data.oro.data
    return (elev / elev.max()).astype(np.float32)

def load_precipitation(shrink, start, stop):
    data = xr.open_mfdataset('pr_surf.ccam_12.5km.nc', combine='by_coords', parallel=True)
    precip = data.pr.sel(time=slice('1980-01-01','2020-12-31'))
    raw = 100 * 21.9587176109 * np.expand_dims(precip[start:stop,:,:].data, axis=3)
    pooled = tf.keras.layers.AveragePooling2D(pool_size=(shrink, shrink), padding='same')(raw)
    return raw, pooled

def match_orography(pq, length):
    pq = np.expand_dims(pq, axis=2)
    return np.repeat(pq[np.newaxis, :, :, :], length, axis=0)
