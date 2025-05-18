import horovod.tensorflow.keras as hvd
import tensorflow as tf
import pickle, yaml, os
from models.SRDN-SO import build_srdcnn_step_orography
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
pq = normalize_orography('oro.nc')

Total = 359424
Epoch_size = 286720
Test_size = Total - Epoch_size
istart = int(hvd.rank()*Epoch_size/hvd.size())
istop = int((hvd.rank()+1)*Epoch_size/hvd.size())
i_test_start = int(hvd.rank()*Test_size/hvd.size()+Epoch_size) + 1
i_test_stop = min(int((hvd.rank()+1)*Test_size/hvd.size()+Epoch_size), Total - 1)

y_train, x_train = load_precipitation('precip', shrink = 8, istart, istop)
y_test, x_test = load_precipitation('precip', shrink = 8, i_test_start, i_test_stop)
p_train = match_orography(pq, x_train.shape[0])
p_test = match_orography(pq, x_test.shape[0])

model = build_srdcnn_step_orography(params['hidden_units'], params['responses'], params['features'],
                                     params['lats'], params['longs'], params['shrink'],
                                     train_cfg['learning_rate'], horovod_opt=hvd.DistributedOptimizer)

callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=10, min_lr=1e-5)
]

if hvd.rank() == 0:
    os.makedirs(data_cfg['model_dir'], exist_ok=True)
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(os.path.join(data_cfg['model_dir'], 'checkpoint-{epoch}.h5'),
                                                        monitor='val_loss', save_best_only=True))

history = model.fit([x_train, p_train], y_train,
                    validation_data=([x_test, p_test], y_test),
                    batch_size=train_cfg['batch_size'],
                    epochs=train_cfg['epochs'], callbacks=callbacks, verbose=2)

if hvd.rank() == 0:
    model.save(os.path.join(data_cfg['model_dir'], 'SRDCNN_STEP_ORO.h5'))
    with open(os.path.join(data_cfg['output_dir'], 'trainHistoryDict.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
