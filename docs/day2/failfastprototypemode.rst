Fail Fast Prototype Mode
========================

When building neural networks, you want things to either work or fail fast.  Long iteration loops are the truest enemy of the  machine learning practitioner.  


To that end, the following techniques will help you out. 

.. code-block:: python

   import torch
   import torch.nn as nn

   # 2dim tensor.. aka a matrix
   x = torch.randn(4, 5)

   # this is the same as:
   batch_size = 4
   feature_size = 5
   x = torch.randn(batch_size, feature_size)

   # now let's try out some NN layer
   output_size = 10
   fc = nn.Linaer(feature_size, output_size)
   print(fc(x).shape)


You can construct whatever prototype variables you want doing this. 

Prototyping an embedding
^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: python

   import torch
   import torch.nn as nn

   batch_size = 4
   sequence_size = 5
   integer_range = 100
   embedding_size = 25
   # notice rand vs randn.  rand is uniform (0,1), and randn is normal (-1,1) 
   random_numbers = (torch.rand(batch_size, sequence_size) * integer_range).long()

   embedder = nn.Embedding(num_embeddings=integer_range, 
                           embedding_dim=embedding_size)

   print(embedder(x).shape)


