Notes for Convolution Models
============================

Implementing a convolutional model can be tricky.  Here are some notes to help get you along:

1. Convolutions in PyTorch expect the channels to be on the 1st dimension
	- We treat the feature vectors (typically from an embedding layer) as the channel / kernel dimension
	- If the shape of your tensor is (batch, seq, feature), then this means a permutation is needed to move the (batch, feature, seq)  
	- To get an intuition as why, imagine an image.  A minibatch of images is (batch, 3, width, height) because an image exists as RGB coordinates for each pixel. 
	- This can also be thought of as 3 feature maps of the image
	- In the same way, the feature dimension of a sequence can be separate feature maps for the entire sequence
	- **The consequence of all of this is a required permute operation post-embedding but pre-convolution: x_embedded.permute(0, 2, 1)**
2.  It is a modeling choice on how to go from the embedded tensor to a final vector, but the goal **is to end up with a single final vector per batch item**
	- This means the channel dimension will be our final vector and we want to apply operations to shrink the sequence dimension until it is size=1
	- You could create enough convolutions that eventually it will shrink to size 1
	- This becomes dependent on the max_seq_len (if this changes, the number of convolutions to shrink to size=1 also changes)
	- Some sort of pooling or lenght-variation operation is recommended for the final reduction. 