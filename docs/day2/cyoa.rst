Choose Your Own Adventure
=========================


The Choose Your Own Adventures have been structured to allow for pure model exploration without worrying as much about the dataset or the training routine.  In each notebook, you will find an implemented dataset loading routine as well as an implemented training routine.  These two parts should seem familiar to you given the last 2 days of content.  What is not implemented is a model definition nor its instantiation.  

It is up to you what you want to use!  Do you want to build the Continuous Bag of Words (CBOW)?  Use an RNN or a CNN?  Do you want to combine the CNN and RNN?  Try out whatever you like and see if you can get the highest accuracy in class!

Strategies for Model Exploration
--------------------------------

Identifying the I/O 
^^^^^^^^^^^^^^^^^^^

A good place to start when doing model exploration is by defining the input-output program that the model is intended to solve.  

If you look at the previous models, you will notice the following pattern:

- Each model starts with embedding the inputs
- Each model ends with applying a Linear layer to create the correct output size

These are the input and output of the models.  

Fail Fast Prototyping
^^^^^^^^^^^^^^^^^^^^^

Use the dataset to get a single batch and the input data from that batch.  You can use that sample input data to prototype an approach to solving the I/O problem.  

.. code-block:: python

   batch = next(iter(DataLoader(dataset, batch_size=4)))
   print(module_to_test(batch['x_data']).shape)

Three simple models to try
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Continuous Bag of Words (CBOW)
	- CBOW is a model which has the following structure: embed the tokens, pool their embeddings in some way, classify the resulting vector.  One way of pooling is to just average them. Others could include taking the max or summing.  A Linear layer is then used to compute the classification vector. 

2. Text Convolutional Neural Network (CNN)
	- CNN will learn spatially invariant patterns because it applies its weights as a sliding window over the input.  You can keep applying more and more CNNs (as in the Chinese Document example), or you could apply one or two and then pool in the same way as the CBOW.  Once you have a single vector for each data point, a final Linear layer is used to compute the classification vector. 

3. Recurrent Neural Network (RNN)
	- Whether the character or word variants, an RNN learns a sequence model of its inputs. In doing classification, the final vector of the sequence is used to represent the entire sequence.  Then, this vector is optionally passed through a couple Linear layers (which themselves can be grouped and described as a Multilayer Perceptron).  Finally, whether the final RNN vector is passed through a Multilayer Perceptron or not, a final Linear layer is used to compute the classification output.  


More complicated models to try
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Better RNNs (Gated Recurrent Unit (GRU), Long Short Term Memory (LSTM))
	- Instead of using the simple RNN provided, use an RNN variant that has gating like GRU or LSTM

2. CNN + RNN
	- One thing you could try is to apply a CNN one or more times to create sequences of vectors that are informed by their neighboring vectors.  Then, apply the RNN to learn a sequence model over these vectors.  You can use the same method to pull out the final vector for each sequence, but with one caveat.  If you apply the CNN in a way that shrinks the sequence dimension, then the indices of the final positions won't quite be right. One way to get around this is to have the CNN keep the sequence dimension the same size.  This is done by setting the `padding` argument to be `kernel_size//2`.  For example, if `kernel_size=3`, then it should be that `padding=1`.  Similarly with `kernel_size=5`, then `padding=2`.  The padding is added onto both sides of the sequence dimension. 

3. Deep Averaging Network
	- The Deep Averaging Network is very similar to CBOW, but has one major differences: it applies an MLP to the pooled vectors. 

4. Using Attention
	- If you're feeling ambitious, try implementing attention! 
	- One way to do attention is use a Linear layer which maps feature vectors to scalars
		+ We begin with a sequence tensor, x_data, that is embedded, x_embedded_sequence = emb(x_data)
		+ The shape here is the similar as the embedded sequence tensor: (batch, sequence, 1)
		+ You can use the apply_across_sequence_loop or apply_across_sequence_reshape  
	- A softmax is then used on the scalar to produce a probability vector
	 	+ attention_weights = F.softmax(attention_weights, dim=2)
	- The probability vector is broadcast (multiplied) across the sequences, so that it weights each sequence vector
		+ weighted_sequence = attention_weights * x_embedded_sequence
	- The sequences are the summed over
		+ weighted_sequence.sum(dim=1)