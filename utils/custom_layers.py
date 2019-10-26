import keras.layers as Kl
import keras.backend as K
from keras.engine.topology import Layer
from keras.activations import softmax
from keras.layers import (Convolution2D, MaxPooling2D, Lambda, Dropout, 
    BatchNormalization, LeakyReLU, Flatten, ZeroPadding2D)
from keras.initializers import glorot_uniform
from utils.seed_util import retrieve_random_state as rrs
import utils.kernel_constraints as k_con
import numpy as np

######################################################################
## ------------------------ CUSTOM LAYERS  ------------------------ ##
######################################################################
   
class GumbelSampler(Layer):
    """ Layer that performs gumbel softmax over the input. 
    """
    def __init__(self
                 , softmax_size
                 , nr_of_samples
                 , tau_init=5.0
                 , anneal_rate=0.01
                 , min_temperature=0.5
                 , samples_per_epoch=128
                 , transpose=False
                 , **kwargs):
        self.tau = K.variable(tau_init, name="temperature")
        self.tau_init = tau_init
        self.softmax_size = softmax_size
        self.nr_of_samples = nr_of_samples
        self.samples_per_epoch = samples_per_epoch
        self.sample_sum = K.variable(1)
        self.epoch_nr = K.variable(1)
        self.anneal_rate = anneal_rate
        self.min_temperature = min_temperature
        self.transpose = transpose
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.batch_size = input_shape[0]
        super().build(input_shape)
    
    # Hacking custom updates during training
    # https://stackoverflow.com/questions/41645990/persistent-variable-in-keras-custom-layer
    def call(self, x):
        # Calling updates
        updates = []
        updates.append((self.sample_sum, self.sample_sum + 1))
        updates.append((self.epoch_nr, self.sample_sum / self.samples_per_epoch))
        updates.append((self.tau, 
                        K.max([self.tau_init * K.exp(- self.anneal_rate * self.epoch_nr), 
                        self.min_temperature])))
        # These updates will be called after each sample.
        self.add_update(updates, x)

        U = K.random_uniform(K.shape(x), 0, 1)
        # Logits + Gumbel noise
        y = x - K.log(-K.log(U + K.epsilon()) + K.epsilon())
        y = softmax(K.reshape(y, (self.batch_size, self.nr_of_samples, self.softmax_size)) / self.tau)
        if self.transpose:
            y = K.permute_dimensions(y, (0, 2, 1))
        return y
    
    def compute_output_shape(self, input_shape):
        if self.transpose:
            output_dim = (input_shape[0], self.softmax_size, self.nr_of_samples)
        else:
            output_dim = (input_shape[0], self.nr_of_samples, self.softmax_size)
        return output_dim

class Duplicate(Layer):
    '''
    Duplicates a tensor along the slice_axis.
    Where nr_of_duplicates indicate how many duplicates there will be made.
    '''
    def __init__(self
                 , nr_of_duplicates
                 , slice_axis=-1 # Inserts the duplicates just after the sliced axis
                 , **kwargs):
        self.nr_of_duplicates = nr_of_duplicates
        self.slice_axis = slice_axis
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.slice_dim = input_shape[self.slice_axis]
        self.input_length = len(input_shape)
        
        # Compute the output dimensions.
        # The layer inserts the "nr_of_duplicates" dimension after the sliced dimension.
        input_shape_list = list(input_shape)
        insert_index = self.slice_axis + 1
        if self.slice_axis < 0:
             insert_index += len(input_shape_list)
        input_shape_list.insert(insert_index, self.nr_of_duplicates)
        self.output_dim = tuple(input_shape_list)
        
        super().build(input_shape)
    
    def call(self, x):
        x_repeat = K.repeat_elements(x, self.nr_of_duplicates, self.slice_axis)
        x_repeat = K.reshape(x_repeat, self.output_dim)
        return x_repeat
    
    def compute_output_shape(self, input_shape):
        return self.output_dim        



class Sum(Layer):
    '''
    Sum the values of a tensor along the given axis.
    '''
    def __init__(self
                 , axis=-1
                 , max_value=False
                 , **kwargs):
        self.axis = axis
        self.max_value = max_value
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.size_sum_dim = input_shape[self.axis]
        input_shape_list = list(input_shape)
        del input_shape_list[self.axis]
        self.output_dim = tuple(input_shape_list)
        self.input_length = len(input_shape)
        super().build(input_shape)
    
    def call(self, x):
        # Note: A prettier implementation can be made using Keras slice
        # Putting all necessary slices in a list using Ellipsis
        x_sum_list = []
        
        for i in range(self.size_sum_dim):
            if self.axis < 0:
                slice_dim = (Ellipsis,) + (i,) + (slice(None),)*(-1*self.axis - 1)
            else:
                slice_dim = (slice(None),)*self.axis + (i,) + (Ellipsis,) 
            x_sum_list.append(x[slice_dim])
        
        # Add all tensors as defined in x_sum_list
        sum_x = Kl.add(x_sum_list)
        
        if self.max_value:
            sum_x = K.minimum(sum_x, K.ones_like(sum_x))
            
        return sum_x
    
    def compute_output_shape(self, input_shape):
        return self.output_dim
        
        
class Min(Layer):
    ''' Removes all values in the tensor that are greater than the given value. 
    '''
    def __init__(self
                 , value=1.0
                 , **kwargs):
        self.value = value
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        super().build(input_shape)
    
    def call(self, x):
        min_x = K.minimum(x, self.value*K.ones_like(x))
        return min_x
    
    def compute_output_shape(self, input_shape):
        return input_shape

######################################################################
## ---------------------- ENCODER FUNCTIONS  ---------------------- ##
######################################################################

def conv_layer(input,
               layer_size,
               kernel_size,
               use_max_pooling=True,
               padding="valid",
               dropout_rate=0.1,
               seed=None):
    '''
    The convolutional layer consisting of:
    Conv layer --> Batch norm --> Dropout (--> Maxpooling)
    '''
    r = rrs(seed)

    x = Convolution2D(layer_size, kernel_size,
                      kernel_initializer=glorot_uniform(seed=r.randint(1e9)),
                      padding=padding)(input)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate, seed=r.randint(1e9))(x)
    if use_max_pooling:
        x = MaxPooling2D((2, 2), padding=padding)(x)
    return x

######################################################################
## ---------------------- DECODER FUNCTIONS  ---------------------- ##
######################################################################

def get_encodings(latent_space, decoder_enc_size, random_seed):
    '''
    Returns a 'dense' layer (using convolutional layers) before the softmax.
    '''
    r = rrs(random_seed)

    # Stamp-selector encoding
    enc_stamps = Convolution2D(decoder_enc_size,
                               (1, 1),
                               activation=LeakyReLU(),
                               kernel_initializer=glorot_uniform(
                                   seed=r.randint(1e9))
                               )(latent_space)
    enc_stamps = BatchNormalization()(enc_stamps)

    # Y-coordinate encoding
    enc_y = Convolution2D(decoder_enc_size,
                          (1, 1),
                          activation=LeakyReLU(),
                          kernel_initializer=glorot_uniform(
                              seed=r.randint(1e9))
                          )(latent_space)
    enc_y = BatchNormalization()(enc_y)

    # X-coordinate encoding
    enc_x = Convolution2D(decoder_enc_size,
                          (1, 1),
                          activation=LeakyReLU(),
                          kernel_initializer=glorot_uniform(
                              seed=r.randint(1e9))
                          )(latent_space)
    enc_x = BatchNormalization()(enc_x)

    return enc_stamps, enc_y, enc_x

def gumbel_softmax_layer(input,
                         softmax_size,
                         gumbel_parameters,
                         nr_of_samples=1,
                         transpose=False,
                         random_seed=False,
                         name="gumbel_sampler"):
    '''
    Perform the gumbel softmax operator.
    '''
    r = rrs(random_seed)

    x = Convolution2D(softmax_size * nr_of_samples, (1, 1),
                      activation=LeakyReLU(),
                      kernel_initializer=glorot_uniform(seed=r.randint(1e9)),
                      padding='same',
                      name="pre-" + name)(input)

    x = BatchNormalization(momentum=0.8)(x)
    x = Flatten()(x)

    soft = GumbelSampler(softmax_size=softmax_size,
                         nr_of_samples=nr_of_samples,
                         tau_init=gumbel_parameters["tau_init"],
                         anneal_rate=gumbel_parameters["anneal_rate"],
                         min_temperature=gumbel_parameters["min_temperature"],
                         samples_per_epoch=gumbel_parameters["steps_per_epoch"],
                         transpose=transpose,
                         name=name)(x)

    return soft


def tensor_product_vector(y_coords, x_coords, side_size):
    '''
    Hardcoded helper function to calculate the Cartesian Product
     or Tensor Product, as there does not exist a TensorDot layer 
     in Keras as of writing.

    We achieve the TensorDot by duplicating the values into a new dimension
     and muliply the tensors.
    '''
    # Duplicates vertically, so coords needs transposing.
    y_coords_dup = Duplicate(
        side_size,
        slice_axis=-2,
        name="duplicator_y")(y_coords)

    # Transpose the Y tensor
    y_coords_dup = Kl.Permute((1, 3, 2))(y_coords_dup)

    x_coords_dup = Duplicate(
        side_size,
        slice_axis=-2,
        name="duplicator_x")(x_coords)

    # Multiplying to obtain single coordinate.
    tensor_product = Kl.multiply(
        [y_coords_dup, x_coords_dup])

    return tensor_product


def tensor_product_matrix_vector(xy_coords, 
                                 stamp_coords, 
                                 nr_of_stamps, 
                                 side_size):
    '''
    Hardcoded helper function to calculate the Cartesian Product
     or Tensor Product, as there does not exist a TensorDot layer 
     in Keras as of writing.

    We achieve the TensorDot by duplicating the values into a new dimension
     and muliply the tensors.
    '''

    # Preparing for multiplication by duplicating the values of the tensor
    #   into a new axis.
    xy_coords_dup = Duplicate(nr_of_stamps, 
                              slice_axis=-3,
                              name="duplicator_coords")(xy_coords)

    xy_coords_dup = Kl.Permute((1, 3, 4, 2))(xy_coords_dup)

    # Preparing for multiplication by duplicating the values of the tensor
    #   into a new axis.
    stamp_coords_dup = Duplicate(side_size,
                                 slice_axis=-2,
                                 name="duplicator_stamps_y")(stamp_coords)
    stamp_coords_dup = Duplicate(side_size,
                                 slice_axis=-2,
                                 name="duplicator_stamps_xy")(stamp_coords_dup)

    xy_stamp_coords = Kl.multiply(
        [xy_coords_dup, stamp_coords_dup])

    return xy_stamp_coords


def stamp_layer(softmax_tensor, stamp_size, random_seed):
    ''' The stamp layer
    
    This layer pads the input according to the size of the stamps and performs
     a convolutional layer. The kernel of this convolutional layer contains
     the stamps.
    '''
    r = rrs(random_seed)

    # Pad the softmax in preparation for the convolution.
    padding_size_left = stamp_size // 2
    padding_size_right = stamp_size // 2 - 1 + stamp_size % 2

    padding_singlet = (padding_size_left, padding_size_right)

    padded_softmax = ZeroPadding2D(padding=(
        padding_singlet, padding_singlet
    ))(softmax_tensor)

    # Place the predicted stamp on the predicted coordinate using
    #  a convolutional operator. 
    # This kernel contains the stamps.
    output = Convolution2D(1, (stamp_size, stamp_size),
                           activation='linear',
                           padding='same',
                           use_bias=False,
                           name='stamp_layer',
                           kernel_constraint=k_con.NonNegMaxOne(),
                           kernel_initializer=glorot_uniform(
        seed=r.randint(1e9))
    )(padded_softmax)

    # Clipping the value to the maximum value of the dataset.
    output = Min(value=1.0)(output)

    return output


def sum_layer(x, axis=-1):
    ''' Performs the sum operation, but only of it's necessary.
    If the dimension equals one, then it will sqeeze the dimension instead.
    '''
    sum_dim = x.shape[axis]

    if sum_dim > 1:
        x = Sum(axis=axis)(x)
    else:
        x = Lambda(lambda x: K.squeeze(x, axis))(x)

    return x