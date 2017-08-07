""" This module contains the sacred experiment
"""

from sacred import Experiment
from sacred.observers import MongoObserver

from mnist_cnn.train import pipeline_train_hook

# create the experiment instance
experiment = Experiment('mnist_demo')

# add a MongoObserver to this experiment
experiment.observers.append(MongoObserver.create(url='dev-mongodb:27017',
                                                 db_name='mnist_demo_experiments'))

# there are several ways to pass the config to sacred
# the most straightforward way is to directly add the config dict explicitly
#
# default_train_spec = generate_train_config(default_args)
# ex.add_config(default_train_spec)
#
# the more 'sacred-onic' (not pythonic) and frankly magical way of doing it is


@experiment.config
def config():
    """ Sacred magically captures the local variables and turns them into a config dict
    """
    name = None
    n_filters = [64, 64, 128, 128]
    nonlinearity = 'relu'
    batch_size = 256
    epoch_size = 256
    n_epochs = 100
    lr = 0.001


# sacred automagically captures arguments in this method, and passes matching objects from the config
@experiment.automain
def main(name, n_filters, nonlinearity, batch_size, epoch_size, n_epochs, lr):
    """ This runs the experiment
    """
    assert name is not None
    pipeline_train_hook(name, n_filters, nonlinearity, batch_size, epoch_size, n_epochs, lr)
