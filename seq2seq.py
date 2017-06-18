import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

PAD=0
EOS=1

number_of_sequences=10
input_sequences=[]
for i in range(number_of_sequences):
	#input_sequences.append(np.random.randint(8,size=random.randint(4,9))+2)
	listt=[]
	for j in range(random.randint(4,9)):
		listt.append(random.randint(2,9))
	input_sequences.append(listt)
#sequences=sequences+2
#output_sequence=input_sequences
longest=len(input_sequences[0])

for i in input_sequences:
	if len(i) > longest:
		longest=len(i)

for i in input_sequences:
	for j in range(longest-len(i)):
		i.append(PAD)
print(longest)
#for i in input_sequences:
#	print i
encoder_input_numpy=np.array(input_sequences)
decoder_targets=input_sequences
for i in decoder_targets:
	i.append(1)
	#print i
decoder_targets_numpy=np.array(decoder_targets)
decoder_inputs=decoder_targets
for i in range(len(decoder_inputs)):
	decoder_inputs[i]=decoder_inputs[i][longest:]+decoder_inputs[i][:longest]

#for i in decoder_inputs:
#	print i

#print(input_sequences)
#print(decoder_inputs)
#print(decoder_targets)
vocab_size=10
input_embedding_size=10

embeddings=tf.Variable(tf.random_uniform([vocab_size,input_embedding_size],-1.0,1.0),dtype=tf.float32)

input_sequences=np.array(input_sequences)
print('Input Sequence : ',encoder_input_numpy.shape)
#print(encoder_input_numpy)
#decoder_targets=np.array(decoder_targets)
print('Decoder Targets : ',decoder_targets_numpy.shape)
#print(decoder_targets_numpy)
decoder_inputs_numpy=np.array(decoder_inputs)
#print(decoder_inputs)
print('Decoder Inputs : ',decoder_inputs_numpy.shape)
#print(decoder_inputs_numpy)

#encoder_input_numpy
#decoder_targets_numpy
#decoder_inputs_numpy

encoder_inputs=tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)
#decoder_targets_embedded

encoder_hidden_units=20
decoder_hidden_units=encoder_hidden_units

encoder_cells=tf.contrib.rnn.LSTMCell(encoder_hidden_units)
encoder_outputs,encoder_final_state=tf.nn.dynamic_rnn(encoder_cells,encoder_inputs_embedded,dtype=tf.float32)

del encoder_outputs

decoder_cell=tf.contrib.rnn.LSTMCell(decoder_hidden_units)
decoder_outputs,decoder_final_state=tf.nn.dynamic_rnn(decoder_cell,decoder_inputs_embedded,initial_state=encoder_final_state,dtype=tf.float32,scope="plain_decoder")

decoder_logits=tf.contrib.layers.linear(decoder_outputs,vocab_size)
decoder_prediction=tf.argmax(decoder_logits,2)
predicting_labels=tf.one_hot(decoder_targets,depth=vocab_size,dtype=tf.float32)
cost=tf.nn.softmax_cross_entropy_with_logits(labels=predicting_labels,logits=decoder_logits)
loss=tf.reduce_mean(cost)
train_op=tf.train.AdamOptimizer().minimize(loss)
hm_epochs=1000

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	flag=0
	epoch_list=[]
	cost_list=[]
	ymaxx=-1
	for i in range(hm_epochs):
		a,b=sess.run([loss,train_op],feed_dict={encoder_inputs:encoder_input_numpy,decoder_inputs:decoder_inputs_numpy,decoder_targets:decoder_targets_numpy})
		if flag == 0 :
			flag=1
			ymaxx=a
		print('Epoch : ',i,' Loss : ',a)
		cost_list.append(a)
		epoch_list.append(i)
		plt.plot(epoch_list,cost_list)
		plt.axis([0,hm_epochs,0,ymaxx])
		plt.title('Cost Graph'+str(i))
		plt.xlabel('Epochs')
		plt.ylabel('Cost values')
		plt.savefig('costimages/'+str(i))
		plt.close()

	print('Original : ',encoder_input_numpy)
	predicted_values=sess.run(decoder_prediction,feed_dict={encoder_inputs:encoder_input_numpy,decoder_inputs:decoder_inputs_numpy,decoder_targets:decoder_targets_numpy})
	print('Predicted : ',predicted_values)
	plt.plot(epoch_list,cost_list)
	plt.show()