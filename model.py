import preproc

# Configure Keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
import gc

# Need to limit GPU memory to 3.5GB given our lovely GTX 970 memory issue.
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = .85
set_session(tf.Session(config=config))

# Train and save model
ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 -1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2)))
model.add(ELU())
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(36,5,5, subsample=(2,2)))
model.add(ELU())
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(48,5,5, subsample=(2,2)))
model.add(ELU())
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(64,3,3, subsample=(1,1)))
model.add(ELU())
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(64,3,3, subsample=(1,1)))
model.add(ELU())
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Flatten())
model.add(ELU())
model.add(Dense(100))
model.add(ELU())
model.add(Dense(50))
model.add(ELU())
model.add(Dense(10))
model.add(ELU())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'] )

train_gen = preproc.train_generator
valid_gen = preproc.validation_generator
train_len = preproc.train_sample_len
valid_len = preproc.valid_sample_len
model.fit_generator(train_gen, samples_per_epoch= train_len, 
                    validation_data=valid_gen, nb_val_samples=valid_len, 
                    verbose=1, nb_epoch=4)
                    
model.save(preproc.modelsave)
del model