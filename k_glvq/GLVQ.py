
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#
import keras
import matplotlib

matplotlib.use('Agg')  # needed to avoid cloud errors

import numpy as np
import argparse
import os

from keras.layers import Input
from keras.layers import Lambda
from keras.models import Model
from keras.datasets import mnist
from keras import callbacks
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from anysma import Capsule
from anysma.capsule import InputModule
from anysma.modules.measuring import OmegaDistance, MinkowskiDistance
from anysma.modules.routing import SqueezeRouting
from anysma.dev import GlvqLoss

sess = K.get_session()


try:
    x_train = np.load('/dataset/x_train.npy')
    x_test = np.load('/dataset/x_test.npy')
    y_train = np.load('/dataset/y_train.npy')
    y_test = np.load('/dataset/y_test.npy')
except:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
y_train = to_categorical(y_train.astype('float32'))
y_test = to_categorical(y_test.astype('float32'))

# network parameters
input_shape = (28, 28, 1)
batch_size = 4096
epochs = 150
glvq_loss = GlvqLoss()

# Define LVQ model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
diss = MinkowskiDistance(linear_factor=None,
                         squared_dissimilarity=True,
                         signal_output='signals')
caps = Capsule(prototype_distribution=(1, 10))
caps.add(InputModule(signal_shape=(-1, np.prod(input_shape)), trainable=False, init_diss_initializer='zeros'))
caps.add(diss)
caps.add(SqueezeRouting())

output = caps(x)
output = Lambda(lambda x: -x)(output[1])

train_model = Model(inputs, output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action='store_true')
    parser.add_argument('--save_dir', default='./output')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_model.summary()

    # Pretrain prototypes over all digits
    print("Pretraining started")
    pre_train_model = Model(inputs=inputs, outputs=diss.input[0])
    diss_input = pre_train_model.predict(x_train, batch_size=batch_size)
    diss.pre_training(diss_input, y_train, capsule_inputs_are_equal=True)

    if args.weights:
        train_model.load_weights(args.weights)

    def acc(y_true, y_pred):
        return metrics.categorical_accuracy(y_true, y_pred)

    def vae_loss(y_true, y_pred):
        loss = glvq_loss(y_true, -y_pred)
        return loss

    train_model.compile(optimizer=Adam(lr=0.01),
                        loss=vae_loss,
                        metrics=[acc])

    def train_generator(x, y, batch_size):
        train_datagen = ImageDataGenerator(width_shift_range=2,
                                           height_shift_range=2,
                                           rotation_range=15)

        generator = train_datagen.flow(x, y, batch_size=batch_size)

        while True:
            batch_x, batch_y = generator.next()
            yield batch_x, batch_y

    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', save_best_only=True,
                                           save_weights_only=True, verbose=1)
    csv_logger = callbacks.CSVLogger(args.save_dir + '/log.csv')
    lr_reduce = callbacks.ReduceLROnPlateau(factor=0.5, monitor='val_loss', mode='min', verbose=1, patience=5)
    # Tensorboard is disabled, as we cant use it anyway
    #tensorboard = TensorBoard(histogram_freq=1, batch_size=batch_size, write_graph=True, write_grads=False,
    #                          write_images=True, log_dir=args.save_dir, images_max_outputs=10)

    train_model.fit_generator(generator=train_generator(x_train, y_train, batch_size),
                              steps_per_epoch=int(y_train.shape[0] / batch_size),
                              epochs=epochs,
                              validation_data=[x_test, y_test],
                              callbacks=[checkpoint, csv_logger, lr_reduce],
                              max_queue_size=40,
                              workers=3,
                              use_multiprocessing=True,
                              verbose=1)

    train_model.save_weights(args.save_dir + '/vae_cnn_mnist.h5')

