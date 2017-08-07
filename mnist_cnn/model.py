""" This module contains methods for constructing the model
"""

from keras.models import Sequential
from keras.layers import Dense

def generate_model_config(name,  n_filters, nonlinearity='relu'):
    """ Generates a spec describing how to build the model

    Args:
      name: name for this model config
      n_filters: list of num filters for each layer - [n_layers]

    Returns:
      model_spec
    """
    model_config = {
        'name': name,
        'n_filters': n_filters,
        'nonlinearity': nonlinearity
    }
    return model_config



def build_model(model_spec):
    """ Build model according to the spec

    Args:
      model_spec
      compile_loss: if True, compile loss
    """
    n_filters = model_spec['n_filters']
    activation = model_spec['nonlinearity']

    # initialize model
    model = Sequential()

    # add first layer, specifying input shape
    model.add(Dense(n_filters[0], activation=activation,
                    input_shape=(None, 784)))

    for n_units in n_filters[1:]:
        model.add(Dense(n_units, activation=activation))

    model.add(Dense(10, activation='softmax'))

    return model
