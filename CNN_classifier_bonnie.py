import keras
import json
import numpy as np
import tensorflow
import matplotlib
import matplotlib.pyplot as plt
from keras.layers import Conv2D, BatchNormalization, Dropout, Activation, Flatten, Dense, MaxPooling2D

from keras.backend.tensorflow_backend import set_session


def load_dataset(filepath):
	mini_speech_data = np.load(filepath)
	train_indices = np.random.permutation(len(mini_speech_data['train_out']))
	train_in = mini_speech_data['train_in'][train_indices]
	train_out = mini_speech_data['train_out'][train_indices]
	test_in = mini_speech_data['test_in']
	test_out = mini_speech_data['test_out']
	labels = mini_speech_data['labels']
	input_shape = train_in[0].shape
	return train_in, train_out, test_in, test_out, input_shape, labels		

def build_CNN(input_shape):
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
	cnn.add(Dense(units=10))
	cnn.add(BatchNormalization())
	cnn.add(Activation(activation=keras.activations.softmax))

	cnn.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(), metrics=['acc'])
	cnn.summary()
	return cnn

def train_CNN(cnn, train_in, train_out, epochs, batchsize, validation_split=0):
	from tensorflow.python.client import device_lib
	device_lib.list_local_devices()
	return cnn.fit(x=train_in,y=train_out,batch_size=batchsize,epochs=epochs,shuffle=True, validation_split=validation_split)

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
	l2, = ax.plot(train_history.history['val_acc'],linestyle='-.')
	l3, = ax.plot(train_history.history['loss'])
	l4, = ax.plot(train_history.history['val_loss'],linestyle='-.')
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
	data_filepath = './mini_speech_data.npz'
	epochs = 50
	batchsize = 100
	train_in, train_out, test_in, test_out, input_shape, labels = load_dataset(data_filepath)
	cnn = build_CNN(input_shape)
	history = train_CNN(cnn, train_in, train_out, epochs=epochs, batchsize=batchsize, validation_split=0.05)
	accuracy = test_CNN(cnn, test_in, test_out, batchsize=batchsize)
	print('CNN test accuracy: {}'.format(accuracy))
	save_CNN(cnn,model_name='./models/speech_CNN_v2',train_history=history,test_accuracy=accuracy)
	load_CNN(model_name='./models/speech_CNN_v2')
	


if __name__ == '__main__':
	main()

