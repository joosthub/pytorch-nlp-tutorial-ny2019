Loading Pretrained Vectors
==========================

It can be extremely useful to make a model which had as advantageous starting point.

To do this, we can set the values of the embedding matrix.


.. code-block:: python

   def get_pretrained_embeddings(filename, dim_size, token_vocab):
       embedding_matrix = torch.zeros(len(token_vocab), dim_size)
       all_words = set(token_vocab.keys())
       
       with open(filename) as fp:
           for line in tqdm_notebook(fp.readlines(), leave=False):
               line = line.split(" ")
               word = line[0]
               if word not in token_vocab:
                   continue
               all_words.remove(word)
               row_index = token_vocab[word]
               embedding_matrix[row_index] = torch.FloatTensor([float(x) for x in line[1:]])
       for remaining_word in all_words:
           row_index = token_vocab[remaining_word]
           embedding_matrix[row_index] = torch.nn.init.kaiming_normal_(torch.zeros(1, dim_size))
               
       return embedding_matrix


Then, we can load that embedding matrix:

.. code-block:: python

   load_pretrained = True
   embedding_size = 32
   pretrained_embeddings = None
   
   if load_pretrained:
       pretrained_embeddings = get_pretrained_embeddings("../data/glove.6B.100d.txt", 
                                                         dim_size=100, 
                                                         token_vocab=dataset.vectorizer.token_vocab)
       embedding_size = pretrained_embeddings.shape[1]


And we can use it in an embedding layer:

.. code-block:: python

   emb = nn.Embedding(embedding_dim=embedding_size, 
                      num_embeddings=num_embeddings, 
                      padding_idx=0, 
                      _weight=pretrained_embeddings)