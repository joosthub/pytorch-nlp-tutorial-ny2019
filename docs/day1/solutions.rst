Solutions 
=========

Problem 1
---------

For when x is a scalar or a vector of length 1:

.. code-block:: python

   def f(x):
       if x > 0:
           return torch.sin(x)
       else:
           return torch.cos(x)

   x = torch.tensor([1.0, 0.5], requires_grad=True)

   y = f(x)
   print(y)
   y.backward()
   print(x.grad)


For when x is a vector, the conditional becomes ambiguous.  To handle this, we can use the python `all` function.  Computing the backward pass requires that the error signal be a scalar.  Since there are now multiple outputs of `f`, we can turn `y` into a scalar just by summing the outputs. 

.. code-block:: python

   def f(x):
       if all(x > 0):
           return torch.sin(x)
       else:
           return torch.cos(x)

   x = torch.tensor([1.0, 0.5], requires_grad=True)

   y = f(x)
   print(y)
   y.sum().backward()
   print(x.grad)


There is one last catch to this: we are forcing the fate of the entire vector on a strong "and" condition (all items must be above 0 or they will all be considered below 0).  To handle things in a more granular level, there are two different methods. 

Method 1: use a for a loop

.. code-block:: python


   def f2(x):
       output = []
       for x_i in x:
           if x_i > 0:
               output.append(torch.sin(x_i))
           else:
               output.append(torch.cos(x_i))
       return torch.stack(output)

   x = torch.tensor([1.0, -1.0], requires_grad=True)
   y = f2(x)
   print(y)
   y.sum().backward()
   print(x.grad)

Method 2: use a mask

.. code-block:: python

   def f3(x):
       mask = (x > 0).float()
       # alternatively, mask = torch.gt(x, 0).float()
       return mask * torch.sin(x) + (1 - mask) * torch.cos(x)

   x = torch.tensor([1.0, -1.0], requires_grad=True)
   y = f3(x)
   print(y)
   y.sum().backward()
   print(x.grad)


Problem 2
---------

.. code-block:: python

   def cbow(phrase):
       words = phrase.split(" ")
       embeddings = []
       for word in words:
           if word in glove.word_to_index:
               embeddings.append(glove.get_embedding(word))
       embeddings = np.stack(embeddings)
       return np.mean(embeddings, axis=0)

   cbow("the dog flew over the moon").shape

   # >> (100,)

   def cbow_sim(phrase1, phrase2):
       vec1 = cbow(phrase1)
       vec2 = cbow(phrase2)
       return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

   cbow_sim("green apple", "green apple")
   # >> 1.0

   cbow_sim("green apple", "apple green")
   # >> 1.0

   cbow_sim("green apple", "red potato")
   # >> 0.749

   cbow_sim("green apple", "green alien")
   # >> 0.683

   cbow_sim("green apple", "blue alien")
   # >> 0.5799815958114477

   cbow_sim("eat an apple", "ingest an apple")
   # >> 0.9304712574359718