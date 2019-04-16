Tensor-Fu-2
===========

Exercise 1
----------


Task: The code below is not quite right for prototyping purposes.  
Fix it so that the indices is more like an actual batched data point
and has batch_size=30 and seq_length=10.

.. code-block:: python

   indices = torch.from_numpy(np.random.randint(0, 100, size=(10,)))

   emb = nn.Embedding(num_embeddings=100, embedding_dim=16)
   assert emb(indices).shape == (30, 10, 16)

Exercise 2
----------

Task: Create a MultiEmbedding class which can input two sets of indices, embed them, and concat the results!

.. code-block:: python

   class MultiEmbedding(nn.Module):
       def __init__(self, num_embeddings1, num_embeddings2, embedding_dim1, embedding_dim2):
           pass

       def forward(self, indices1, indices2):
           # use something like
           # z = torch.cat([x, y], dim=1)

           pass


   # testing

   # use indices method from above
   # the batch dimensions should agree
   # indices1 = 
   # indices2 = 
   # multiemb = MutliEmbedding(num_emb1, num_emb2, size_emb1, size_emb2)
   # output = multiemb(indices1, indices2)
   # print(output.shape) # should be (batch, size_emb1 + size_emb2)