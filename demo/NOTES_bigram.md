# WordSmith notes

## Bigram model

### Data

- present inside the `/data` dir and contains a .txt file 

- Number of samples -> of `6474`

- smallest length ->  `3`  | largest length -> `19`

- Total pairs that can be founded are `47,617`


### Bigram character level inference

- One character is followed by other and so one word has many training samples.

- We group the samples in groups of consecutive sequesnce (bigrams) including the start and end characters.

- We then basically just count how often the second character will follow the first.

- for example, (e, m), (m, m), (m, a) for emma so characters `m` and `a` are very likely to follow `m`. High probabiilty.

- We store the pairs couts in a 2D array. where, row is first val and column in second val in pair so the cell contains the r,c pair count

- Visualize this using matplotlib for frequency

- Put the `'.'` index as 0 since it's easier. Now basically if you want to know which word should come after f, just take the `f` row and see the probability distribution.

- To normalize the row to get probabilities, we take the row and divide individual element by the sum of the entire row. (This much value out of all of this value)

- `torch.multinomaial()` to sample from the probability distribution. Basically give me probs and I will give you integers that are sampled from this prob

- torch.Generator() to fix the ramdomness

### Tensor Shapes (Broadcasting)

- `tensor.sum()` gets an int or tuple of ints as input dim. This is basically `across` which dimension to sum.

Example -> If the tensor.shape is (27, 27) then if I do `tensor.sum(0)` that means i am telling take the `0th` dimention (in this case row) and sum across it so this. This will result in a single column tensor becase we summed across rows and it `squeezed out` the dim across what we summed.
The resulting tensor will habe shape (27) (Now this is just a row vector even though it should be a column vector). Which only has one dimension.

Additionally we can do `keep_dims = True` to maintain the shape and structure while summing. If i do `tensor.sum(0, keep_dims = True)` then it will sum across the 0th dim (rows) like before and not squeeze out the dim. So the resultant shape is (1, 27).

** So basically we summed vertically each colum. Resulting in one row and same columns.

Therefore, we want to sum across the columns and keep the dim to basically sum horizontally. We will do `tensor.sum(1, keep_dims = True)`

- Now to do, `p = p / tensor.sum(1, keep_dim = True)`, dividing [27, 27] by [27, 1]

- Here are the broadcasting rules
- 1) Each tensor should have atleast one dimension
- 2) Iterating over the dimensions from trailing side, either both should be equal, one should be `1` or one does not exist. Then and only then torch will broadcast it.

### Loss function (Likelihood)

- Likelihood is basically the product of the output probabilities and higher the probabilities, higher the likelihood number and better the model is performing (confidence)
- Since numbers are between 1 and 1 the product is very small and for convenience we use log of the likelihood which gives a reasonable number.
- log(1) is `zero` and log(0) is infinity. So basically perfect model will get log likelihood of zero. And worse models will have negative values so we take negative log likelihood to get a positive number. Higher means big loss, lower is less loss.

`torch.log(prob)`

log(prob_a*prob_b*prob_c) = log(prob_a) + log(prob_b) + log(prob_c)
So we can keep accumulating log probs than multiplying all probs.

Gaol -> Maximize the likelihood of the data wrt model parameters.
that means maximizing log likelihood
that means minimizing the neative log likelihood
that means minimizing average negative log likelihood

### Smoothing out the prob distribution

- How likely is the model to predict the name `tejasq`.
nll is infinity. Because prob of `s` followed by `q` is zero hence log (0) -> ininity.

- We can smooth out the probability distribution by adding `+1` to all probs. This will smooth the distribution a little. Similar to softmax with temperature. If you add +1000 to everything, all values in the row will be almost quual and hence even distribution.

## MLP character level inference

- Create xs and ys dataset where x is ch1 and y is ch2 in the bigram

### One hot encoding

import torch,nn.functional as F
hot_xs = F.one_hot(xs, num_classes = 27)        #shape [rows of xs, 27]

So now each row is a training example of 27 features


### Neural net weight initialize

- Initializing weights -> torch.randn() will initialize random weights from a normal dist of mean 0 and standard deviation 1 (Standard Normal Distrubution)

- Single input shape will be [1, 27] (just one row of 27 hot encoding)
- Batch of 32 input shape will be [32, 27].

- We initialize the weights for these neurons to be of shape [27, 1] <b>(#inputs, #outputs)</b>. (Single column with rows as many as features or hot encodings).

- When performed matrix product, [32, 27] @ [27, 1] result is a [32, 1].

- This Neuron basically takes 27 inputs (one hots) and outputs one value. These [32, 1] outputs are 32 activations. One for each activation.

- We want 27 outputs for prob of each char so we put 27 neurons. So weight is initialized of shape [27, 27] [#inputs, #outputs]

Example -> output[3, 13] represents activation of the 13th neuron when the 3rd example data was passed

### Logits (log-counts)

- Logits are basically the raw outputs from the network (before softmax or probs)
- These can be negative or positive and hence we exponent it for convinience in this case. Because exp converts negative number to between 0 an 1 and positive numbers to > 1
- Then normalize by sum. Eventually doing softmax operation.


### Loss

`loss = -probs[torch.arange(len(xs)), ys].log().mean()`

- Probs[x, y] is basically in xth example yth activation
- We take all the examples's activation and then log it and mean it and negate it to get negative log likelihood loss.


### One Hot Neural nets

- when we pass a one hot input to newral net that is fully connected, if the 5th element is 1.0 and other zero, we basically just taking out 5th row of the weight matrix.


- If the weight matrix is all initialized to zero, the output is a perfectly normal distribution (smoothing out the prob dist)
- Adding a weight component to the loss function will increase the loss. And this will then try to squeeze the weight closer to zero...the more closer to zero, the more smooth the dist is and less the overfitting is.
- We can choose the regularization strength by regularization rate