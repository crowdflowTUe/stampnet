# Tensorflow & GPU settings
from utils.custom_layers import *
from utils.seed_util import retrieve_random_state as rrs
import utils.img_gen as ig

# Keras imports
import keras
from keras.objectives import mse
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model
from keras.activations import softmax
import keras.backend as K
from keras.layers import Input

# Other useful imports
import os
import tensorflow as tf
import numpy as np

# Sacred imports
from sacred.observers import FileStorageObserver
from sacred import Experiment


# Setting up GPU environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set correct fuzzfactor for Keras.
K.set_floatx('float32')
K.set_epsilon(np.finfo(np.float32).eps)

# Starting the Sacred experiment and adding an observer that collects stats.
# These will be stored runs/Deep autoencoder localization 2, fixed shapes/...
# Files that are created are:
# - 'run.json', collecting various information about dependencies and
#               computer environment.
# - 'config.json', all variables and random seed number.
# - 'cout.txt', output of the text printed to the termiinal.
ex = Experiment()
ex.observers.append(
    FileStorageObserver.create(
        basedir=os.path.join('runs', ex.path)))

### IMPORTANT NOTE: this file does not work without a configuration file 
###  that stores all different hyperparameters used in the experiment.
### We load these hyperparameters from a different experiment file
###  to avoid code duplication.
###
### However, these variables can be added inline using the following function:
### @ex.config
### def config():
###     # Add hyper parameters here.


######################################################################
## --------------------------- ENCODER  --------------------------- ##
######################################################################
@ex.capture
def encoder(input, encoder_parameters, random_seed):

    r = rrs(random_seed)

    encoder_dropout_rate = encoder_parameters["encoder_dropout_rate"]
    kernel_size = encoder_parameters["conv_kernel_size"]

    x = input

    # Encoding the features of the input using convolutional layers.
    # The convolutional layer consist of:
    # Convolution --> BatchNorm --> Dropout --> Maxpooling
    for layer_size in encoder_parameters["conv_layer_sizes"]:
        x = conv_layer(x, layer_size, kernel_size,
                       dropout_rate=encoder_dropout_rate,
                       padding="same",
                       seed=r.randint(1e9))

    final_conv_size = encoder_parameters["final_conv_size"]
    final_conv_kernel = encoder_parameters["final_conv_kernel"]

    # Latent space
    # Make sure that we end up with a 1x1 latent space.
    latent_space = conv_layer(x, final_conv_size, final_conv_kernel,
                              use_max_pooling=False,
                              padding="valid",
                              dropout_rate=encoder_dropout_rate,
                              seed=r.randint(1e9))

    return latent_space


######################################################################
## --------------------------- DECODER  --------------------------- ##
######################################################################


@ex.capture
def decoder(latent_space, decoder_parameters, random_seed):
    r = rrs(random_seed)

    enc_stamps, enc_y, enc_x = get_encodings(
        latent_space, decoder_parameters["decoder_enc_size"], r)

    coord_size = decoder_parameters["coord_tensor_size"]
    nr_of_stamps = decoder_parameters["nr_of_stamps"]
    stamps_per_canvas = decoder_parameters["stamps_per_canvas"]
    gumbel_parameters = decoder_parameters["gumbel_parameters"]

    y_coords = gumbel_softmax_layer(
        input=enc_y,
        softmax_size=coord_size,
        gumbel_parameters=gumbel_parameters,
        nr_of_samples=stamps_per_canvas,
        name="softmax_y",
        random_seed=r.randint(1e9)
    )

    x_coords = gumbel_softmax_layer(
        input=enc_x,
        softmax_size=coord_size,
        gumbel_parameters=gumbel_parameters,
        nr_of_samples=stamps_per_canvas,
        name="softmax_x",
        random_seed=r.randint(1e9)
    )

    stamp_coords = gumbel_softmax_layer(
        input=enc_stamps,
        softmax_size=nr_of_stamps,
        gumbel_parameters=gumbel_parameters,
        nr_of_samples=stamps_per_canvas,
        name="softmax_stamp",
        random_seed=r.randint(1e9)
    )

    # Calculate the tensor / cartesian product of the X and Y coordinates.
    # (Multiplies all combinations of X and Y coords.)
    xy_coords = tensor_product_vector(y_coords, x_coords, coord_size)

    # Calculate the tensor of the coordinates and the stamp selector.
    xy_stamp_coords = tensor_product_matrix_vector(
        xy_coords,
        stamp_coords,
        nr_of_stamps,
        coord_size)

    # Sum all different predicted stamp coordinates into a single image.
    xy_stamp_coords_sum = sum_layer(xy_stamp_coords, axis=-4)

    # The stamp layer with different stamps.
    decoded = stamp_layer(xy_stamp_coords_sum,
                          decoder_parameters["stamp_size"],
                          random_seed=r.randint(1e9))

    return decoded, stamp_coords, xy_coords


######################################################################
## ---------------------- MODEL DEFINITION  ----------------------- ##
######################################################################
@ex.capture
def model(layer_parameters, random_seed, compile=True):

    r = rrs(random_seed)

    input_parameters = layer_parameters["input_parameters"]

    input = Input(
        batch_shape=(input_parameters['batch_size'],
                     input_parameters['img_rows'],
                     input_parameters['img_cols'],
                     input_parameters['img_channels']),
        dtype='float32',
        name='main_input'
    )

    latent_space = encoder(input,
                           layer_parameters["encoder_parameters"],
                           random_seed=r.randint(1e9))

    decoded, stamp_coords, xy_coords = decoder(
        latent_space,
        layer_parameters["decoder_parameters"],
        random_seed=r.randint(1e9))

    # Create the autoencoder with inputs and three different outputs.
    # The final two are used to analyze the softmax.
    autoencoder = Model(input, [decoded, stamp_coords, xy_coords])
    # generator = decoder

    # Use sum of squared errors as a loss function.
    def my_loss(x, x_hat):
        recon = K.sum(mse(x, x_hat), axis=(-1, -2))
        return recon

    if compile:
        autoencoder.compile(optimizer='nadam',
                            loss=my_loss,
                            loss_weights=[1., 0., 0.])

    return autoencoder


######################################################################
## ---------------------------- MAIN  ----------------------------- ##
######################################################################
@ex.automain
def main(run_parameters, layer_parameters, _seed=None):
    # Generating a different reproducible random number from the seed
    # The second one is used for the image data generation.
    r1 = rrs(_seed)
    r2 = rrs(r1.randint(1e9))

    epochs = run_parameters["epochs"]
    steps_per_epoch = run_parameters["steps_per_epoch"]
    validation_steps = run_parameters["validation_steps"]

    autoencoder = model(layer_parameters, random_seed=_seed)

    # Retrieve the output shapes for the input generator
    # Also converts the tensorflow Dimension into integers
    output_shapes = [[int(j) for j in tensor.shape]
                     for tensor in autoencoder.outputs]

    # Retrieve the data for the input.
    data = ig.retrieve_data(
        output_shapes,
        img_type=run_parameters["img_type"],
        img_type_parameters=run_parameters["img_type_parameters"],
        shuffle=run_parameters["shuffle"],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        random_seed=r2)

    # Create & set the paths and directories for saving weights & Tensorboard.
    train_dir = os.path.join(ex.observers[0].dir, "trained")
    tensorboard_dir = os.path.join(ex.observers[0].dir, 'board')
    weights_path = os.path.join(
        train_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")

    # Creating the "trained" directory if it doesn't exist.
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    # Checkpointer for saving the weights during training.
    checkpointer = keras.callbacks.ModelCheckpoint(
        weights_path,
        monitor='val_loss',
        verbose=0,
        save_best_only=False,
        save_weights_only=True,
        mode='auto',
        period=10
    )

    autoencoder.fit_generator(
        data['training_generator'],
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=data['val_generator'],
        validation_steps=validation_steps,
        verbose=2,
        max_queue_size=20,
        callbacks=[
            checkpointer,
            TensorBoard(log_dir=tensorboard_dir)],
        workers=1
    )
