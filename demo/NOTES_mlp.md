# MLP Notes

- The original paper predicts the next words given the previous words. But we are going to implement the same model on a character-level. Using an MLP

- They have a volaculary of 17,000 words and each word is embedded as a 30 dimension vector

- They are maximizing the log-likelihood of the training data.

Example -> If the model has never seen the sentence `"A dod was running in the ______"`
And during test time it is trying to predict this word but it has never seen this example before. This is called `being out of distribution`. But maybe it has seen the sentence "`The cat is running in the _____"` and the model has leared the place the embeddings of `The` and `A` very closely. So you can still generalize bu this approach.

## Overview

- Takes in 3 characters to predict the next one
- Each character is represented as an integer. 'a'->1, 'z'->25, '.'->0
- We have a lookup table C (embedding weights) of size [vocab, emb_dim] so in this case [27, 10]. Each characted denoted by 10 dimension embeddings and we have vocab of 17 character.
- We fetch the embeddings for 'a' by plucking up that row from the lookup table.
- So now we have [3, 10] table of inputs of 3 words
- This will be passed to a hidden layer bottleneck
- And that will be passed to the output layer with as many neurons as vocab. In our case 27.
- Apply softmax to get prob of next character. Pluck out the correct output's prob and maximize it's likelyhood by finding loss.
- Update the loss based on all weights.


## One hot encoding and lookup table

- If we have a lookup table of shape [vocab, #emb_dim] in our case [27, 2]
- To embed number 5 or `e` we do just take 5th row like C[5]
- But instead, we could also make one-hot encoding of number 5 with 27 classes [27] where 5th bit is tuened on
- And Multiply it with the C matrix that will eventually pluck out the 5th row only.
- [1, 27]x[27, 2] -> [1, 2] which is the fifth row

## Pytorch Indexing

- We can index from dimension like C[3] will give 3rd row if it has 2 dimension or give 3rd value if it was one dimension only.
- We can also index with list or tensors like, C[[2, 3, 4]] this will give those respextive rows or values
- We can also index with multi-dimension tensors
Examlple -> X is [10,000, 3] C is [27, 2] so doing C[X] will actually return a tensor of shape [10,000, 3, 2] so all those [10,000, 2] now have 2 values in depth. Like a cube.

## Pytoch tools

- 1) torch.cat() -> Concatenates a sequence of torches `across` a dimension you give. Example, `torch.cat(([90, 3], [90, 3]), 1)` will give [90, 6]

- 2) torch.unbind() -> Unbindes (removes) tensors across a dimension and returns a squence of torches. (Kinda Opposite of concatenate)
Example: torch.unbind([3, 2, 5], 1) will give ([3, 0, 4], [3, 1, 4], [3, 2, 4])

- 3) torch.view() (Very efficient). Views in a given shape without changing it in memory.
Example -> torch.arange(18) is shaped [18] if I do a.view((9,2)) it will be a 9,2 shaped tensor.

## Cross Entropy Loss

Cross entropy is used for classification problems. It takes in logits and targets as inputs.
Internally it calculates the softmax, to get probabilities and then takes the correct prob value we want,
it logs it, and calculate the mean and negating it. So basically calculating negative log likelihood dirextly from logits and targets. Much efficient.


## Logits trick

- Very high value of logit like 100 will create a problem. For example. while doing softmax, exp(100) will give inf because it's too large of a number. This will result in a nan in the probablity.
- Cross_Entroppy handles all of this internally.
- <b>How???</b>
- You can add any value to all the logits and the softmax won't change. Think about it. so adding -100 to everything will negate this error.


## Overfitting a single Batch first

- It is a good practice to overfit the single batch data data to see if everything is woking including loss accumulation

## Finding a good learning rate

- Figure out the lower bound and upper bound by trial and error. By seeing if it is going down smooth for low end and unstable for high end
- Create a list of learning rates from lower bound to upper bound.
- INcrease it in every epoch and plot all rates and epochs and see where it losses stability

# Finding if the model is good

- Split the dataset into train/dev/test
- After training on train set, got  loss of 2.3 witht slight noise due to batch size
- After feedforwarding on dev set, got a loss of 2.29
- Therefore, the model has not even memorized the train set yet. It's underfitting.
- 