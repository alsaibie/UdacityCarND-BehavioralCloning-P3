import preprocess

# Configure Keras
from keras.models import Sequential
from keras.models import load_model
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

# Helper function from https://github.com/fchollet/keras/issues/2640
def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False


model_file = 'model.h5'
# Train and save model
ch, row, col = 3, 160, 320  # Trimmed image format

model = load_model(model_file)
layers_to_pop = 9
for _ in range(layers_to_pop):
	model.pop()
# Freeze remaining bottom layers
for layer in model.layers:
	layer.trainable = False

# define top model. TODO: Do I need to run this separately first and save weights? Or have them zeroed first?
# Wait. Could we just have frozen the weights for the first x layers and then fine tuned further?



top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(ELU())
top_model.add(Dense(100))
top_model.add(ELU())
top_model.add(Dense(50))
top_model.add(ELU())
top_model.add(Dense(10))
top_model.add(ELU())
top_model.add(Dense(1))

model.add(top_model)

model.compile(loss='mse', optimizer='adam',  metrics=['accuracy'] )
print(model.summary())
train_gen = preprocess.train_generator
valid_gen = preprocess.validation_generator
train_len = preprocess.train_sample_len
valid_len = preprocess.valid_sample_len
model.fit_generator(train_gen, samples_per_epoch= train_len, 
                    validation_data=valid_gen, nb_val_samples=valid_len, 
                    verbose=1, nb_epoch=3)
                    
                    
model.save('model7.h5')