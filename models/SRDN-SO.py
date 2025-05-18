import tensorflow as tf
from tensorflow.keras import layers, regularizers
import horovod.tensorflow.keras as hvd

def SRDN_STEP_ORO(numHiddenUnits, numResponses, numFeatures, numLats, numLongs, shrink):
    """
    Builds a super-resolution downscaling CNN model using Conv2DTranspose layers with Orography input.

    Parameters:
    - numHiddenUnits: Number of filters in hidden conv layers.
    - numResponses: Number of output channels.
    - numFeatures: Number of input feature channels.
    - numLats: Latitude dimension of high-res data.
    - numLongs: Longitude dimension of high-res data.
    - shrink: Downsampling factor for inputs.

    Returns:
    - A compiled Keras model with Horovod distributed optimizer.
    """
    reg_val = 1e-9
    reg = regularizers.l2(reg_val)

    # Inputs
    input_shape = (numLats // shrink, numLongs // shrink, numFeatures)
    inputs = layers.Input(shape=input_shape, name='low_res_input')
    orog = layers.Input(shape=(numLats, numLongs, numFeatures), name='orography_input')

    # First deconvolution + orography merge
    x = layers.Conv2DTranspose(numHiddenUnits, (7, 7), strides=2, activation='relu', padding='same',
                               kernel_regularizer=reg, bias_regularizer=reg)(inputs)
    
    orog_1 = layers.AveragePooling2D(pool_size=4, padding='same')(orog)
    x = layers.Concatenate(axis=-1)([x, orog_1])

    # Second deconvolution + orography merge
    x = layers.Conv2DTranspose(numHiddenUnits, (7, 7), strides=2, activation='relu', padding='same',
                               kernel_regularizer=reg, bias_regularizer=reg)(x)
    
    orog_2 = layers.AveragePooling2D(pool_size=2, padding='same')(orog)
    x = layers.Concatenate(axis=-1)([x, orog_2])

    # Final upsampling to original resolution
    x = layers.Conv2DTranspose(numHiddenUnits, (7, 7), strides=2, activation='relu', padding='same',
                               kernel_regularizer=reg, bias_regularizer=reg)(x)
    x = layers.Conv2D(numResponses, (1, 1), activation='linear', padding='same',
                      kernel_regularizer=reg, bias_regularizer=reg)(x)

    # Merge with original orography again and refine
    x = layers.Concatenate(axis=-1)([x, orog])
    x = layers.Conv2D(numHiddenUnits, (7, 7), activation='relu', padding='same',
                      kernel_regularizer=reg, bias_regularizer=reg)(x)
    outputs = layers.Conv2D(numResponses, (1, 1), activation='linear', padding='same',
                            kernel_regularizer=reg, bias_regularizer=reg)(x)

    # Build and compile model
    model = tf.keras.models.Model(inputs=[inputs, orog], outputs=outputs)

    # Horovod: adjust learning rate based on number of GPUs.
    opt = tf.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999,
               epsilon=None, decay=0.0, amsgrad=False)
    opt = hvd.DistributedOptimizer(opt)

    model.compile(loss='mse', optimizer=opt, metrics=['msle','mae'], experimental_run_tf_function=False)

    print( model.summary() ) if hvd.rank() == 0 else None

    return model
