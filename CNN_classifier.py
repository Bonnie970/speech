import os
import keras
import json
import numpy as np
import tensorflow
import matplotlib
import matplotlib.pyplot as plt
from keras.layers import Conv2D, BatchNormalization, Dropout, Activation, Flatten, Dense, MaxPooling2D


class ValBest(keras.callbacks.Callback):
    def __init__(self, monitor="val_loss", verbose=0, mode="auto", period=1):
        super(ValBest, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.epochs_since_last_save = 0

        if(mode not in ["auto", "min", "max"]):
            mode = "auto"

        if(mode == "min"):
            self.monitor_op = np.less
            self.best = np.Inf
        elif(mode == "max"):
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if("acc" in self.monitor):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_train_begin(self, logs=None):
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if(self.epochs_since_last_save >= self.period):
            self.epochs_since_last_save = 0
            current = logs.get(self.monitor)
            if(self.monitor_op(current, self.best)):
                if(self.verbose > 0):
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f, saving weights' % (epoch + 1, self.monitor, self.best, current))
                self.best = current
                self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)

def load_dataset(filepath):
    mini_speech_data = np.load(filepath)
    train_indices = np.random.permutation(len(mini_speech_data['train_out']))
    train_in = mini_speech_data['train_in'][train_indices]
    train_out = mini_speech_data['train_out'][train_indices]
    test_in = mini_speech_data['test_in']
    test_out = mini_speech_data['test_out']
    labels = mini_speech_data['labels']
    input_shape = train_in[0].shape
    num_classes = len(labels)
    print(num_classes)
    return train_in, train_out, test_in, test_out, input_shape, labels, num_classes

def build_CNN(input_shape,num_classes):
    cnn = keras.models.Sequential()
    # conv layer 1
    cnn.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', strides=(1,1), input_shape=input_shape))
    cnn.add(BatchNormalization())
    cnn.add(Activation(activation=keras.activations.relu))
    # max pooling
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    # conv layer 2
    cnn.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', strides=(1,1)))
    cnn.add(BatchNormalization())
    cnn.add(Activation(activation=keras.activations.relu))
    # max pooling
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    # conv layer 3
    cnn.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', strides=(1,1)))
    cnn.add(BatchNormalization())
    cnn.add(Activation(activation=keras.activations.relu))
    # max pooling
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    # conv layer 4
    cnn.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', strides=(1,1)))
    cnn.add(BatchNormalization())
    cnn.add(Activation(activation=keras.activations.relu))
    # max pooling
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    cnn.add(Flatten())
    # fc layer 1
    cnn.add(Dense(units=128))
    cnn.add(Activation(activation=keras.activations.relu))
    cnn.add(Dropout(rate=0.2))
    # fc layer 2
    cnn.add(Dense(units=64))
    cnn.add(Activation(activation=keras.activations.relu))
    cnn.add(Dropout(rate=0.2))
    # output layer
    cnn.add(Dense(units=num_classes))
    cnn.add(BatchNormalization())
    cnn.add(Activation(activation=keras.activations.softmax))

    cnn.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(), metrics=['acc'])
    cnn.summary()
    return cnn

def train_CNN(cnn, train_in, train_out, epochs, batchsize, validation_split=0):
    from tensorflow.python.client import device_lib
    device_lib.list_local_devices()
    if validation_split>0:
        callback = ValBest()
    else
        callback = None
    return cnn.fit(x=train_in, y=train_out, batch_size=batchsize,\
                   epochs=epochs, shuffle=True, callbacks=[callback],\
                   validation_split=validation_split)

def test_CNN(cnn, test_in, test_out, batchsize):
    acc = cnn.evaluate(x=test_in,y=test_out,batch_size=batchsize)
    return acc[1]

def save_CNN(cnn,model_name,train_history,test_accuracy):
    # save json
    model_json = cnn.to_json()
    with open(model_name+'.json','w') as json_file:
        json_file.write(model_json)
        json_file.close()
    # save weight
    cnn.save_weights(model_name+'.h5')
    # save train_history graph
    fig, ax = plt.subplots()
    l1, = ax.plot(train_history.history['acc'])
    l2, = ax.plot(train_history.history['val_acc'])
    l3, = ax.plot(train_history.history['loss'])
    l4, = ax.plot(train_history.history['val_loss'])
    ax.grid(True)
    ax.set_xlabel('epochs')
    ax.set_ylabel('accuracy')
    ax.set_title('{} final accuracy: {:.4f}'.format(model_name,test_accuracy))
    ax.legend((l1,l2,l3,l4),('train accuracy','val accuracy','train loss','val loss'))
    plt.savefig(model_name+'.png')
    print('model saved to disk')

def load_CNN(model_name):
    json_file = open(model_name+'.json','r')
    model_json = json_file.read()
    json_file.close()
    cnn = keras.models.model_from_json(model_json)
    cnn.load_weights(model_name+'.h5')
    cnn.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(), metrics=['acc'])
    cnn.summary()
    print('model loaded successfully')
    return cnn

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    data_filepath = './mini_speech_data.npz'
    # training setting
    epochs = 50
    batchsize = 100
    # read dataset
    train_in, train_out, test_in, test_out, input_shape, labels, num_classes = load_dataset(data_filepath)
    print(train_in.shape)
    print(input_shape)
    # build model
    cnn = build_CNN(input_shape,num_classes)
    # train and tes # train and testt
    history = train_CNN(cnn, train_in, train_out, epochs=epochs, batchsize=batchsize, validation_split=0.05)
    accuracy = test_CNN(cnn, test_in, test_out, batchsize=batchsize)
    print('CNN test accuracy: {}'.format(accuracy))

    save_CNN(cnn,model_name='./models/speech_CNN_mini_v2',train_history=history,test_accuracy=accuracy)
    load_CNN(model_name='./models/speech_CNN_mini_v2')
    return


if __name__ == '__main__':
    main()

