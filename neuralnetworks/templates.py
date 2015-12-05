import numpy as np
import pandas as pd
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn.metrics import classification_report, confusion_matrix

#constants
# PIXELS = 48

def BasicNN(max_epochs=10, hidden_num_units=20, update_learning_rate=0.01, input_shape=(None, 32), output_num_units=18):
	net = NeuralNet(
	    layers=[
	        ('input', layers.InputLayer),
	        ('hidden', layers.DenseLayer),
	        ('output', layers.DenseLayer)
	    ],
	    input_shape=input_shape,

	    hidden_num_units=hidden_num_units,

	    output_nonlinearity=lasagne.nonlinearities.softmax,
	    output_num_units=output_num_units,

	    #optimization parameters
	    update=nesterov_momentum,
	    update_learning_rate=update_learning_rate,
	    max_epochs=max_epochs,
	    verbose=10000
	)

	return net

def BasicNN2(max_epochs=10, hidden = [20], update_learning_rate=0.01, input_shape=(None, 32), output_num_units=18):
	
	NN_layers = [ ('input', layers.InputLayer) ]
	hidden_args = {}
	for i, num_units in enumerate(hidden):
		NN_layers.append( ('hidden'+str(i) , layers.DenseLayer) )
		hidden_args['hidden'+str(i)+'_num_units'] = num_units

	NN_layers.append( ('output', layers.DenseLayer) )


	net = NeuralNet(
	    layers=NN_layers,
	    input_shape=input_shape,

	    output_nonlinearity=lasagne.nonlinearities.softmax,
	    output_num_units=output_num_units,

	    #optimization parameters
	    update=nesterov_momentum,
	    update_learning_rate=update_learning_rate,
	    max_epochs=max_epochs,
	    verbose=10000,
	    **hidden_args
	)

	return net

# def DoubleCNN(learning_rate=0.01, max_epochs=100, update_learning_rate=0.01, input_shape=(None, 1, PIXELS, PIXELS)):
# 	net = NeuralNet(
# 		layers=[
# 			('input', layers.InputLayer),
# 			('conv2d1', layers.Conv2DLayer),
# 			('maxpool1', layers.MaxPool2DLayer),
# 			# ('dropout1', layers.DropoutLayer),
# 			('conv2d2', layers.Conv2DLayer),
# 			('maxpool2', layers.MaxPool2DLayer),

# 			('dense1', layers.DenseLayer),
# 			('dropout2', layers.DropoutLayer),

# 			('output', layers.DenseLayer)
# 		],
# 		# input layer descriptors
# 		input_shape=(None, 1, PIXELS, PIXELS),


# 		# convolution layer descriptors
# 		conv2d1_num_filters=16,
# 		conv2d1_filter_size=(5,5),
# 		conv2d1_nonlinearity = lasagne.nonlinearities.rectify,
# 		conv2d1_W = lasagne.init.GlorotUniform(),

# 		# maxppol layer descriptors
# 		maxpool1_pool_size=(2,2),

# 		# dropout layer descriptors
# 		#     dropout1_p = 0.5,

# 		# convolution layer descriptors
# 		conv2d2_num_filters=32,
# 		conv2d2_filter_size=(3,3),
# 		conv2d2_nonlinearity = lasagne.nonlinearities.rectify,
# 		conv2d2_W = lasagne.init.GlorotUniform(),

# 		# maxppol layer descriptors
# 		maxpool2_pool_size=(2,2),

# 		dense1_num_units=64,


# 		# dropout layer descriptors
# 		dropout2_p = 0.5,

# 		# output layer descriptors
# 		output_nonlinearity = lasagne.nonlinearities.softmax,
# 		output_num_units=10,

# 		#optimization parameters
# 		update=nesterov_momentum,
# 		update_learning_rate=learning_rate,
# 		max_epochs=max_epochs,
# 		verbose=1000000
# 	)

# 	return net


# def TripleCNN(learning_rate=0.01, max_epochs=100, update_learning_rate=0.01, input_shape=(None, 1, PIXELS, PIXELS)):
# 	net = NeuralNet(
# 		layers=[
# 			('input', layers.InputLayer),
# 			('conv2d1', layers.Conv2DLayer),
# 			('maxpool1', layers.MaxPool2DLayer),
# 			# ('dropout1', layers.DropoutLayer),
# 			('conv2d2', layers.Conv2DLayer),
# 			('maxpool2', layers.MaxPool2DLayer),
# 			('conv2d3', layers.Conv2DLayer),
# 			('maxpool3', layers.MaxPool2DLayer),
# 			('dense1', layers.DenseLayer),
# 			('dropout2', layers.DropoutLayer),

# 			('output', layers.DenseLayer)
# 		],
# 		# input layer descriptors
# 		input_shape=(None, 1, PIXELS, PIXELS),


# 		# convolution layer descriptors
# 		conv2d1_num_filters=16,
# 		conv2d1_filter_size=(5,5),
# 		conv2d1_nonlinearity = lasagne.nonlinearities.rectify,
# 		conv2d1_W = lasagne.init.GlorotUniform(),

# 		# maxppol layer descriptors
# 		maxpool1_pool_size=(2,2),

# 		# dropout layer descriptors
# 		#     dropout1_p = 0.5,

# 		# convolution layer descriptors
# 		conv2d2_num_filters=32,
# 		conv2d2_filter_size=(3,3),
# 		conv2d2_nonlinearity = lasagne.nonlinearities.rectify,
# 		conv2d2_W = lasagne.init.GlorotUniform(),

# 		# maxppol layer descriptors
# 		maxpool2_pool_size=(2,2),

# 		# convolution layer descriptors
# 		conv2d3_num_filters=64,
# 		conv2d3_filter_size=(2,2),
# 		conv2d3_nonlinearity = lasagne.nonlinearities.rectify,
# 		conv2d3_W = lasagne.init.GlorotUniform(),

# 		# maxppol layer descriptors
# 		maxpool3_pool_size=(2,2),

# 		dense1_num_units=64,


# 		# dropout layer descriptors
# 		dropout2_p = 0.5,

# 		# output layer descriptors
# 		output_nonlinearity = lasagne.nonlinearities.softmax,
# 		output_num_units=10,

# 		#optimization parameters
# 		update=nesterov_momentum,
# 		update_learning_rate=learning_rate,
# 		max_epochs=max_epochs,
# 		verbose=1000000
# 	)

# 	return net

# def QuadNN(learning_rate=0.01, max_epochs=100, update_learning_rate=0.01, input_shape=(None, 1, PIXELS, PIXELS)):
# 	net = NeuralNet(
# 		layers=[
# 			('input', layers.InputLayer),
# 			('conv2d1', layers.Conv2DLayer),
# 			('maxpool1', layers.MaxPool2DLayer),
# 			# ('dropout1', layers.DropoutLayer),
# 			('conv2d2', layers.Conv2DLayer),
# 			('maxpool2', layers.MaxPool2DLayer),
# 			('conv2d3', layers.Conv2DLayer),
# 			('maxpool3', layers.MaxPool2DLayer),

# 			('conv2d4', layers.Conv2DLayer),
# 			('maxpool4', layers.MaxPool2DLayer),

# 			('dense1', layers.DenseLayer),
# 			('dropout2', layers.DropoutLayer),
# 			('output', layers.DenseLayer)
# 		],
# 		# input layer descriptors
# 		input_shape=(None, 1, PIXELS, PIXELS),


# 		# convolution layer descriptors
# 		conv2d1_num_filters=16,
# 		conv2d1_filter_size=(5,5),
# 		conv2d1_nonlinearity = lasagne.nonlinearities.rectify,
# 		conv2d1_W = lasagne.init.GlorotUniform(),

# 		# maxppol layer descriptors
# 		maxpool1_pool_size=(2,2),

# 		# dropout

# 		# dropout1_p = 0.5,

# 		# convolution layer descriptors
# 		conv2d2_num_filters=32,
# 		conv2d2_filter_size=(3,3),
# 		conv2d2_nonlinearity = lasagne.nonlinearities.rectify,
# 		conv2d2_W = lasagne.init.GlorotUniform(),

# 		# maxppol layer descriptors
# 		maxpool2_pool_size=(2,2),

# 		# convolution layer descriptors
# 		conv2d3_num_filters=64,
# 		conv2d3_filter_size=(3,3),
# 		conv2d3_nonlinearity = lasagne.nonlinearities.rectify,
# 		conv2d3_W = lasagne.init.GlorotUniform(),

# 		maxpool3_pool_size=(2,2),




# 		# convolution layer descriptors
# 		conv2d4_num_filters=128,
# 		conv2d4_filter_size=(3,3),
# 		conv2d4_nonlinearity = lasagne.nonlinearities.rectify,
# 		conv2d4_W = lasagne.init.GlorotUniform(),

# 		# maxppol layer descriptors
# 		maxpool4_pool_size=(2,2),

# 		dense1_num_units=128,

# 		# dropout layer descriptors
# 		dropout2_p = 0.5,

# 		# output layer descriptors
# 		output_nonlinearity = lasagne.nonlinearities.softmax,
# 		output_num_units=10,

# 		#optimization parameters
# 		update=nesterov_momentum,
# 		update_learning_rate=learning_rate,
# 		max_epochs=max_epochs,
# 		verbose=1000000
# 	)


