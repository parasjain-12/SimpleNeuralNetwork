import numpy as np

#Input array
X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])

#Output
y=np.array([[1],[1],[0]])

#ReLU Function
def ReLU (x):
	return x * (x>0)

#Derivative of ReLU Function
def derivatives_ReLU(x):
	return 1 * ( x > 0)

#Variable initialization
epoch=1000 #Setting training iterations
lr=0.01 #Setting learning rate
inputlayer_neurons = X.shape[1] #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))

for i in range(epoch):

	#Forward Propogation
	hidden_layer_input1=np.dot(X,wh)
	hidden_layer_input=hidden_layer_input1 + bh
	hiddenlayer_activations = ReLU(hidden_layer_input)
	output_layer_input1=np.dot(hiddenlayer_activations,wout)
	output_layer_input= output_layer_input1+ bout
	output = ReLU(output_layer_input)

	#Backpropagation
	E = y-output
	slope_output_layer = derivatives_ReLU(output)
	slope_hidden_layer = derivatives_ReLU(hiddenlayer_activations)
	d_output = E * slope_output_layer
	Error_at_hidden_layer = d_output.dot(wout.T)
	d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
	wout += hiddenlayer_activations.T.dot(d_output) *lr
	bout += np.sum(d_output, axis=0,keepdims=True) *lr
	wh += X.T.dot(d_hiddenlayer) *lr
	bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr

print (output)