import keras
import json
import numpy as np
import tensorflow
import matplotlib
import matplotlib.pyplot as plt
from keras.layers import Conv2D, BatchNormalization, Dropout, Activation, Flatten, Dense, MaxPooling2D

def load_dataset(filepath):
	mini_speech_data = np.load(filepath)
	train_in = mini_speech_data['train_in']
	train_out = mini_speech_data['train_out']
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

def train_CNN(cnn, train_in, train_out, epochs, batchsize):
	from tensorflow.python.client import device_lib
	device_lib.list_local_devices()
	cnn.fit(x=train_in,y=train_out,batch_size=batchsize,epochs=epochs,shuffle=True)

def test_CNN(cnn, test_in, test_out, batchsize):
	acc = cnn.evaluate(x=test_in,y=test_out,batch_size=batchsize)
	return acc[1]

def save_CNN(cnn,json_filepath,weight_filepath):
	model_json = cnn.to_json()
	with open(json_filepath,'w') as json_file:
		json_file.write(model_json)
		json_file.close()
	cnn.save_weights(weight_filepath)
	print('model saved to disk')

def load_CNN(json_filepath,weight_filepath=None):
	json_file = open(json_filepath,'r')
	model_json = json_file.read()
	json_file.close()
	cnn = keras.models.model_from_json(model_json)
	if weight_filepath!=None:
		cnn.load_weights(weight_filepath)
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
	train_CNN(cnn, train_in, train_out, epochs=epochs, batchsize=batchsize)
	test_CNN(cnn, test_in, test_out, batchsize=batchsize)
	save_CNN(cnn,json_filepath='./models/speech_CNN_v1.txt',weight_filepath='./models/speech_CNN_v1.h5')
	cnn2 = load_CNN(json_filepath='./models/speech_CNN_v1.txt',weight_filepath='./models/speech_CNN_v1.h5')


if __name__ == '__main__':
	main()

