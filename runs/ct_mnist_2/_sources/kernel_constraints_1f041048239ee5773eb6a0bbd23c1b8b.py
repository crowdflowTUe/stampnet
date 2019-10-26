from keras.constraints import non_neg, min_max_norm, Constraint
import keras.backend as K

class NonNegMinMaxNorm(Constraint):
    """MinMaxNorm weight constraint.

    Constrains the weights incident to each hidden unit
    to have the norm between a lower bound and an upper bound.

    # Arguments
        min_value: the minimum norm for the incoming weights.
        max_value: the maximum norm for the incoming weights.
        rate: rate for enforcing the constraint: weights will be
            rescaled to yield
            `(1 - rate) * norm + rate * norm.clip(min_value, max_value)`.
            Effectively, this means that rate=1.0 stands for strict
            enforcement of the constraint, while rate<1.0 means that
            weights will be rescaled at each step to slowly move
            towards a value inside the desired interval.
        axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Conv2D` layer with `data_format="channels_last"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
    """

    def __init__(self, min_value=0.0, max_value=1.0, rate=1.0, axis=0):
        self.min_value = min_value
        self.max_value = max_value
        self.rate = rate
        self.axis = axis

    def __call__(self, w):
        norms = K.sqrt(K.sum(K.square(w), axis=self.axis, keepdims=True))
        desired = (self.rate * K.clip(norms, self.min_value, self.max_value) +
                (1 - self.rate) * norms)
        w *= (desired / (K.epsilon() + norms))
        
        # Ensure non-negativity constraint.
        w *= K.cast(K.greater_equal(w, 0.), K.floatx())
        return w

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value,
                'rate': self.rate,
                'axis': self.axis}

class NonNegMaxOne(Constraint):
    """NonNegMaxOne weight constraint
    
    Constrains the weights to be non-negative and not above 1.
    """

    def __call__(self, w):
        w *= K.cast(K.greater_equal(w, 0.), K.floatx())
        
        w = K.minimum(w, K.ones_like(w))
        # w *= K.cast(K.less_equal(w, 1.), K.floatx()) + K.minimum(K.cast(K.greater(w, 1.0), K.floatx()), K.ones_like(w))
        return w

class ClipWeights(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, min_value=0.0, max_value=1.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'min_value': self.min_value,
                'max_value': self.max_value}

    
    
class MaxAbsOne(Constraint):
    """MaxAbsOne weight constraint
    
    Constrains the weights to be non-negative and not above 1.
    """

    def __call__(self, w):
        w = K.maximum(w, -K.ones_like(w))
        w = K.minimum(w, K.ones_like(w))
        # w *= K.cast(K.less_equal(w, 1.), K.floatx()) + K.minimum(K.cast(K.greater(w, 1.0), K.floatx()), K.ones_like(w))
        return w
    
# Alias
NonNegNonPos = NonNegMaxOne

class ConstrainedMinMaxNorm(Constraint):
    """MinMaxNorm weight constraint.

    Constrains the weights incident to each hidden unit
    to have the norm between a lower bound and an upper bound.

    # Arguments
        min_value: the minimum norm for the incoming weights.
        max_value: the maximum norm for the incoming weights.
        rate: rate for enforcing the constraint: weights will be
            rescaled to yield
            `(1 - rate) * norm + rate * norm.clip(min_value, max_value)`.
            Effectively, this means that rate=1.0 stands for strict
            enforcement of the constraint, while rate<1.0 means that
            weights will be rescaled at each step to slowly move
            towards a value inside the desired interval.
        axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Conv2D` layer with `data_format="channels_last"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
    """

    def __init__(self, min_value=0.0, max_value=1.0, rate=1.0, axis=0):
        self.min_value = min_value
        self.max_value = max_value
        self.rate = rate
        self.axis = axis

    def __call__(self, w):
        # Ensure no value above 1.
        w = K.minimum(w, K.ones_like(w))

        # Ensure non-negativity constraint.
        w *= K.cast(K.greater_equal(w, 0.), K.floatx())


        norms = K.sqrt(K.sum(K.square(w), axis=self.axis, keepdims=True))
        desired = (self.rate * K.clip(norms, self.min_value, self.max_value) +
                (1 - self.rate) * norms)
        w *= (desired / (K.epsilon() + norms))
        



        return w

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value,
                'rate': self.rate,
                'axis': self.axis}



