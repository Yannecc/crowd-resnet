from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model, Sequential


from batch_maker import StimMaker, all_test_shapes

from tensorflow.keras.layers import Dense, Input, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.utils import to_categorical


########################################################################################################################
# Decoder Layer Generator
########################################################################################################################

def Minimodel(n_hidden, input_shape):
    mod = Sequential()
    mod.add(Flatten(input_shape=input_shape[1:]))
    mod.add(Dense(n_hidden, activation='elu'))
    mod.add(Dense(2, activation='softmax'))

    return mod

########################################################################################################################
# Training
########################################################################################################################


def train_loop(base_model, sample_layers, input_maker, train_n_batches, batch_size,  NAME, ID_MODEL, lr=1e-6):
    print('\rTraining...')
    print('\rModel ID:'+NAME+str(ID_MODEL))

    path = './logdir' + NAME + str(ID_MODEL)

    train_summary_writer = tf.summary.create_file_writer(path)

    minimodels = [Minimodel(n_hidden, base_model.layers[L].output_shape) for L in sample_layers]

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = tf.keras.metrics.SparseCategoricalAccuracy()

    with train_summary_writer.as_default():


        optimizer = Adam(lr=lr)

        for iteration in range(train_n_batches):
            print('Batch n°'+str(iteration) + ' / ' + str(train_n_batches))

            #Generate train data:
            batch_data, batch_labels = input_maker.generate_Batch(batch_size, [0, 0, 1, 0], noiseLevel=.1,
                                                              normalize=False,
                                                              fixed_position=None)
            RES_outputs = base_model(batch_data)

            for i, mini in enumerate(minimodels):
                with tf.GradientTape() as tape:
                    logits = mini(RES_outputs[i])
                    loss = loss_object(batch_labels, logits)
                    gradients = tape.gradient(loss, mini.trainable_variables)
                optimizer.apply_gradients(zip(gradients, mini.trainable_variables))
                tf.summary.scalar('___loss'+str(i), loss, step=optimizer.iterations)

                acc = metrics(batch_labels, logits)
                metrics.reset_states()
                if iteration % 25 == 0:
                    print('Decoder n°'+str(i)+' training accuracy:'+str(acc.numpy()*100)+'%')
                tf.summary.scalar('___acc' + str(i), acc, step=optimizer.iterations)



        # Saving the minimodels
        for i, mini in enumerate(minimodels):
            for layer in mini.layers:
                layer.trainable = False
            mini.save(path + '/Minimodel_L='+str(sample_layers[i])+'.h5')


########################################################################################################################
# Testing
########################################################################################################################





def test_loop(base_model, sample_layers, input_maker, SHAPES, test_set_size,  NAME, ID_MODEL):
    print('\rTesting...')

    path= './logdir' + NAME + str(ID_MODEL)

    #Recovering saved models
    minimodels = [tf.keras.models.load_model(path+'/Minimodel_L='+str(L)+'.h5') for L in sample_layers]

    N_tests = len(SHAPES)

    results = np.zeros((N_tests, len(sample_layers)))


    metrics = tf.keras.metrics.SparseCategoricalAccuracy()

    for s,shapematrix in enumerate(SHAPES):

        print('\rshapematrix ={}'.format(str(shapematrix)))
        batch_data, batch_labels = input_maker.generate_Batch(test_set_size, [0, 0, 0, 1], noiseLevel=.1,
                                                                  normalize=False,
                                                                  fixed_position=None, shapeMatrix=shapematrix)
        RES_outputs = base_model(batch_data)

        for i, mini in enumerate(minimodels):
            logits = mini(RES_outputs[i])
            metrics.reset_states()
            results[s,i] = metrics(batch_labels, logits)

    print('Finished testing for this model, numpy saving\n'.format())
    np.save(path + '/results', results)
    np.save(path + '/shape_list', SHAPES)

    return results


########################################################################################################################
#Main:
########################################################################################################################
IMG_SIZE = 224 # 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
n_models = 10
NAME = 'ResNet_test'

#Training parameters:
train_n_batches = 10000
batch_size = 64


test_set_size = 64*100
n_hidden = 512

### BASE MODEL DEFINITION (UNIQUE):
# activation layers before next conv : 4, 38, 80, 142, 174
sample_layers = [4, 38, 80, 142, 174]

model = ResNet50(include_top=False, weights='imagenet', input_shape=IMG_SHAPE)
outputs = [model.get_layer(model.layers[L].name).output for L in sample_layers]

intermediate_layer_model = Model(inputs=model.input, outputs=outputs)

# input definition
input_maker = StimMaker(imSize=(IMG_SIZE, IMG_SIZE), shapeSize=19, barWidth=2)
# test shapes definition
SHAPES = all_test_shapes()


### MAIN LOOP
for m in range(n_models):
    ### Model is defined and saved via train_loop functions
    ## Uniquely identified by the NAME and MODEL_ID
    ID_MODEL = m
    train_loop(intermediate_layer_model, sample_layers, input_maker, train_n_batches, batch_size, NAME, ID_MODEL)
    test_loop(intermediate_layer_model, sample_layers, input_maker, SHAPES, test_set_size,  NAME, ID_MODEL)


