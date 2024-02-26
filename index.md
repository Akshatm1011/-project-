# Enhancing NanoGPT: Squentropy Loss and Hyperparameter Tuning!

## Can optimizing the squentropy loss function and adjusting pertinent hyperparameters improve baseline NanoGPT performance?

**Introduction**

Large Language Models have typically been trained on the 
cross-entropy loss function due to its effectiveness 
in classification properties. However, the benefits of using 
regression-based loss functions should not be underestimated
when it comes to performance. Our project proposes to combine 
the benefits that come with regression-based loss functions with 
the practicality of classification-based loss functions to create
a more robust loss function and see if the performance of the model
can be improved. In our case we will be using mean squared error loss
in conjunction with cross-entropy loss to create Squentropy.

**Mathematical Foundations**
Squentropy loss is a hybrid loss function that combines aspects of cross entropy and mean squared error.  
        
Consider the following notation:

Let $D = \{(x_1, y_1), ..., (x_n, y_n)\}$ denote the dataset sampled from a joint distribution $D(X , Y)$. 
For each sample $i,  x_i \in X$ is the input and $y_i \in Y = \{1, 2, . . . , C\}$ is the true class label. The one-hot encoding label used for training is $e_{y_{i}} = [0, ..., 1, ..., 0] \in \mathbb{R}^C$. 
Let $f(x_i) \in \mathbb{R}$ denote the logits (output of last linear layer) of a neural network of input $x_i$, with components $f_j(x_i)$, $j = \{1, 2, . . . , C\}$.
Let $p_{i,j} = \frac{e^{f_j(x_i)}}{\sum_{j=1}^C f_{j}(x_{i})^{2}}$ denote the predicted probability of $x_i$ to be in class $j$.  

Then the squentropy loss function on a single sample $x_i$
is defined as follows:

$$
\begin{equation*}
    L_{squen}(x_{i}, y_{i})  = -\log p_{i, y_{i}}(x_{i}) + \frac{1}{C - 1}{ \sum_{j=1, j \neq y_{i}}^C f_{j}(x_{i})^{2}}
\end{equation*}
$$

The squared loss portion of $L_{squen}$ acts as a $\textit{regularization}$ term.


**Methods**

We first started with simply using Mean Squared error in place of
cross-entropy to evaluate our model's performance. As expected
this failed to output the same intelligibility in sentences
that was the case with our cross-entropy trained model.
We then attempted to reward the correctly identified logit
when calculating our mean squared error through multiplying
it by 100 so that our training would be more effective.
This seemed to improve our intelligibility in sentence structure
however, it too did not compare to the results of the 
cross entropy trained model. Finally, we combined the losses of
mean squared error and cross-entropy to get the best results
so far in terms of intelligibility but are in the process of 
tuning hyperparameters to increase intelligibility.


**Hyperparameter Tuning**

Andrej Kaparthy (creator of the NanoGPT repo)
states that the current set of hyperparameters utilized 
are have not been tuned for optimal performance! 
We have decided to focus on the learning
rate, dropout percentage, and number of layers in
the neural network. Here are their potential values:  
• Lr - 3  

• Batch Size - 10  

• max_iter - 100  

**Cross Entropy Baseline**

The standard loss function used in NLP token 
prediction scenarios is cross-entropy. Our GPT-2 model
trained on Tiny Stories performed well on simple
cross-entropy without any hyper-paramter tuning.
The model converged with around 1.8 loss value
starting at around 10. The perplexity of the model
is around 3.8 for the baseline using simple cross-
entropy. The sampled text output using cross entropy was
coherent sentences that made sense grammatically
and plot wise. There was a initialization and a 
conflict as well as an ending/cliamx. The cross entropy
loss allowed the model to pick up the nuances in the
training data and text.

**Main Result**

Since we were able to train NanoGPT with a novel loss
function not designed for NLP Tasks, we can say
that MSE, Squentropy, and other novel loss func-
tions could be potential avenues to explore for an
LLM. Moreover, the adaptability of these models to
alternative loss functions opens up new research 
directions, potentially leading to more robust language
understanding and generation capabilities. As seen
in the Squentropy Best Performance section, we were
able to match the performance of cross entropy to
squentropy (with a perplexity of 3.1). This aligns
with our hypothesis that alternative loss functions
can not only match but potentially exceed the performance of 
traditional loss functions in specific scenarios, 
suggesting a promising area for further investigation 
and application in the field of natural language processing.

**Generalization**

Traditionally speaking, Cross entropy has been the
dominant loss function used within the domain of
large language models. This is due to the nature
of our model being trained on language and capturing 
the nuances within it as opposed to simply
finding how close outputs are to a true value. Thus,
classification loss functions are used over regression
based ones since language, and capturing the dynamics 
in large text, must be classified rather than
purely quantified. However, given the benefits that
are present in regression loss functions, such as 
punishing values that deviate from the mean (i.e., Mean
Squared Error), it is believed that adding said 
benefits to the pre-existing loss function may improve
the model’s training. Given that the hyperparameters 
aren’t tuned on the data specifically but rather
the overall performance of the model, they would
need to be retuned to allow for the full effects of the
Squentropy loss function to be seen.

**Conclusions and Outlook**

Since the perplexity of the Squentropy trained model
was ... this indicates that although our new model
comes close to the results of a Cross-entropy trained
model, it does not supersede them. In addition, the
text output of the Squentropy model was a lot less
intelligible than that of the Cross-entropy model.
The future implications of our findings, although not
exactly superseding the results compared to cross 
entropy, show that there may be future loss functions
combined with cross entropy that could yield similar
results or improve upon our findings to generate a
more accurate loss function than cross entropy itself.

**References**

[1] Hui, Mikhail, Belkin, M., & Wright, S., Cut your Losses
with Squentropy, arXiv:2302.03952 [cs.LG] (2023).
Available at: https://doi.org/10.48550/arXiv.2302.03952.
[2] Hui, Mikhail, & Belkin, M., Evaluation of Neural
Architectures Trained with Square Loss vs
Cross-Entropy in Classification Tasks,
arXiv:2006.07322 [cs.LG] (2021). An extended version
published at ICLR2021 with added evaluations of
Transformer architectures. Available at:
https://arxiv.org/abs/2006.07322.
