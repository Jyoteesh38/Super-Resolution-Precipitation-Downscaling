import horovod.tensorflow.keras as hvd
import tensorflow as tf
import pickle, yaml, os
from models.SRDN_SO import SRDN_STEP_ORO
from models.utils import normalize_orography, load_precipitation, match_orography
#----------------------------------------------------------------------------------------------
# Horovod: initialize Horovod.
#----------------------------------------------------------------------------------------------
hvd.init()

print ('***hvd.size ', hvd.size(),' hvd.rank', hvd.rank(), 'hvd.local_rank() ', hvd.local_rank())

# Horovod: pin GPU to be used to process local rank (one GPU per process)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
print(' gpus = ', gpus)
if hvd.local_rank() == 0:
    print("Socket and len gpus = ",socket.gethostname(), len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
#----------------------------------------------------------------------------------------------
# Set total number of images, training epoch size, test data size
#
#  - must be Scalable/divisible to multi nodes to ensure load balance

# 41 years of data 1980-2020📍️

Total_images = 359424
Epoch_size = 286720
Test_size  = Total_images - Epoch_size

# Batch size - aim to fill GPU memory to achieve best computational performance
batch_size = 16
#----------------------------------------------------------------------------------------------
# Horovod: Split the test data across multiple processors   
#----------------------------------------------------------------------------------------------
istart = int(hvd.rank()*Epoch_size/hvd.size())
istop  = int((hvd.rank()+1)*Epoch_size/hvd.size())

i_test_start = int(hvd.rank()*Test_size/hvd.size()+Epoch_size) + 1
i_test_stop  = int((hvd.rank()+1)*Test_size/hvd.size()+Epoch_size)

if i_test_stop >= Total_images:
  i_test_stop = Total_images - 1

print ( '*** rank = ', hvd.rank(),' istart = ', istart, ' istop = ', istop)
print ( '*** rank = ', hvd.rank(),' i_test_start = ', i_test_start, ' i_test_stop = ', i_test_stop)
#----------------------------------------------------------------------------------------------
# Set key parameters
#----------------------------------------------------------------------------------------------
numHiddenUnits = 64
numResponses = 1
numFeatures  = 1
shrink = 8
numLats = 512 
numLongs = 512
#----------------------------------------------------------------------------------------------
# Data loading
#----------------------------------------------------------------------------------------------
pq = normalize_orography('oro.nc')
y_train, x_train = load_precipitation('precip', shrink, istart, istop)
y_test, x_test = load_precipitation('precip', shrink, i_test_start, i_test_stop)
p_train = match_orography(pq, x_train.shape[0])
p_test = match_orography(pq, x_test.shape[0])
#----------------------------------------------------------------------------------------------
# Horovod: create callbacks required for horovod model run
#----------------------------------------------------------------------------------------------
callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),

    # Reduce the learning rate if training plateaues.
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, monitor='val_loss', mode='min', patience=10, min_lr=0.00001, verbose=1),
#    dropout_scheduler
    # LMS(swapout_threshold=1, swapin_groupby=0, swapin_ahead=1), # These are the max swapping, slowest data throughput parameters. Adding sync_mode=3 would also allow for higher amount of data.
]
#----------------------------------------------------------------------------------------------
# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
#----------------------------------------------------------------------------------------------
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5', monitor='val_loss', save_best_only=True))
#----------------------------------------------------------------------------------------------
# 	Build the model
#----------------------------------------------------------------------------------------------
model = SRDCNN_STEP_ORO(numHiddenUnits, numResponses, numFeatures, numLats, numLongs, shrink)
#----------------------------------------------------------------------------------------------
# 	Train the model
#----------------------------------------------------------------------------------------------
# Setup timer for training step
t0[hvd.rank()] = time.time()
# Add a barrier to sync all processes before starting training
hvd.allreduce([0], name="Barrier")
print ('*** rank = ', hvd.rank(),' Train model')
history = model.fit([x_train, p_train], y_train, callbacks=callbacks, epochs=100, verbose=2, 
                      validation_data = ([x_test, p_test], y_test)) 
# Elapsed time for training operation
elapsed_time = time.time() - t0[hvd.rank()]
print ('*** rank = ', hvd.rank(),' Total Training Elapsed Time (sec) = ', elapsed_time)
#for history in histories:
with open('./trainHistoryDict_{}_GPU'.format(hvd.size()), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
if hvd.rank() == 0:
   model.save('./SRDCNN_STEP_ORO.h5')
