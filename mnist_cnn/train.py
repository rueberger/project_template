""" This module contains the pipeline train method
"""

from mnist_cnn.model import build_model
from mnist_cnn.batch_gen import pipeline_batch_gen

from keras.optimizers import Adam

def generate_train_config(model_spec, train_batch_spec, val_batch_spec,  epoch_size,
                        n_epochs=100, lr=0.001):
    """ Generates a spec fully specifying a run of the experiment

    Args:
      model_spec: a model spec
      train_batch_spec: a batch spec for train
      val_batch_spec: batch spec for validation
      epoch_size: int
      n_epochs: int
      lr: learning rate - float

    Returns:
      train_spec
    """

    train_spec = {
        'model_spec': model_spec,
        'train_batch_spec': train_batch_spec,
        'val_batch_spec':  val_batch_spec,
        'train_spec': {
            'epoch_size': epoch_size,
            'n_epochs': n_epochs,
            'lr': lr
        }
    }
    return train_spec

def pipeline_train(train_spec):
    """ Pipelined train method that trains model according to the spec
    """
    model = build_model(train_spec['model_spec'])
    # compile model loss
    optimizer = Adam(lr=train_spec['lr'])
    model.compile(loss='categorical_cross_entropy', optimizer=optimizer)

    train_batch_gen = pipeline_batch_gen(train_spec['train_batch_spec'])
    val_batch_gen = pipeline_batch_gen(train_spec['val_batch_spec'])


    model.fit_generator(train_batch_gen, train_spec['epoch_size'], n_epochs=train_spec['n_epochs'],
                        validation_data=val_batch_gen, validation_steps=128)
