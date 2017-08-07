""" This module contains the sacred experiment
"""

from sacred import Experiment
from sacred.observers import MongoObserver

from mnist_cnn.model import build_model
from mnist_cnn.batch_gen import pipeline_batch_gen


from keras.optimizers import Adam

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
    model_spec = {
        name: None,
        n_filters: [64, 128, 128],
        nonlinearity: 'relu'
    }
    train_batch_spec = {
        mode: 'train',
        batch_size: 32
    }
    val_batch_spec = {
        mode: 'val',
        batch_size: 32
    }
    train_spec = {
        epoch_size: 100,
        n_epochs: 100,
        lr: 0.0001
    }

# sacred automagically captures arguments in this method, and passes matching objects from the config
@experiment.automain
def main(train_spec, model_spec, train_batch_spec, val_batch_spec, epoch_size,
          n_epochs=100, lr=0.001):
    """ This runs the experiment
    """
    model = build_model(train_spec['model_spec'])
    # compile model loss
    optimizer = Adam(lr=train_spec['lr'])
    model.compile(loss='categorical_cross_entropy', optimizer=optimizer)

    train_batch_gen = pipeline_batch_gen(train_spec['train_batch_spec'])
    val_batch_gen = pipeline_batch_gen(train_spec['val_batch_spec'])


    model.fit_generator(train_batch_gen, train_spec['epoch_size'], n_epochs=train_spec['n_epochs'],
                        validation_data=val_batch_gen, validation_steps=128)
