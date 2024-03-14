<center>
<sub>Group Members:</sub> <br>
<sub>Akshat Muir (akmuir@ucsd.edu), Sujay Talanki (stalanki@ucsd.edu), Rehan Ali (rmali@ucsd.edu), Sujen Kancherla (skancherla@ucsd.edu)</sub> <br>
<sub>Project Mentors: Misha Belkin (mbelkin@ucsd.edu), Yian Ma (yianma@ucsd.edu)</sub>
</center>

**Introduction**

In the context of LLM’s, there has been a growing interest in improving the performance
of compact models such as NanoGPT. These models not only aim to generate coherent text
but also strive to optimize resource utilization in text generation tasks. We are interested in
optimizing NanoGPT’s performance; we will particularly focus on the loss function that the
model attempts to minimize and hyperparameters that can be tuned. In machine learning,
a model makes a prediction by choosing the input that minimizes a loss function. LLM’s
traditionally use cross entropy Loss function, primarily because it is well-suited for tasks
that generate probabilistic predictions (and classification tasks in general). The MSE loss
function is typically utilized for regression tasks (It uses the residuals- the error between
the predicted value and the actual value.). However, there is a way to utilize the MSE loss
function for our application. It involves predicting the next token in the sequence (choosing
the token with the maximum probability of occurring) and comparing this token to the
actual token. These residuals will be computed through vector algebra, and inputted into
the loss function to compute a final metric. The goal is to implement this mathematical
transformation into code, and evaluate the result on our dataset to see if the model performs
better. This research aims to contribute to both the advancement of NanoGPT and other
large language models in the field of natural language processing.

**Mathematical Foundations**  

Squentropy loss is a hybrid loss function that combines aspects of cross entropy and mean squared error.  
        
Consider the following notation:

![](latex.png)

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
so far in terms of intelligibility. We have also hyper-parametered
tuned our model to be even more intelligibile.

**Hyperparameter Tuning**

LLM performance can be improved by choosing the optimal hyperparameters in the training
process (via hyperparameter tuning). For our purposes, we chose to change the learning
rate, number of layers in the model, and the dropout rate. Using our hyperparameter tuning
script, we implemented a grid search: exhaustively search all possible combinations
of hyperparameter values within our defined search space.

Search space:

learning_rates = [0.000006, 0.0006, 0.06]
dropouts = [0.0, 0.1, 0.2]
n_layers = [8, 12, 16]

We chose the model with the hyperparameters that resulted in the lowest perplexity metric.
The most optimal hyperparameters found for squentropy are written below.

• Lr - 0.00006

• Number of Layers - 16

• dropout - 0.1

**Script For Tuning**
```python
import os
from itertools import product

learning_rates = [0.000006,0.0006,0.06]
dropouts = [0.0, 0.1, 0.2]
n_layers = [8, 12, 16]

params = list(product(learning_rates, dropouts, n_layers))
path = os.getcwd().split('/')[:3]
path += ['teams', 'b13', 'group1']
out = os.path.join(*path)

counter = 0
for lr, dropout, n_layer in params:
    command = f'python3 train.py --compile=False --wandb_log=True --out_dir={out} --batch_size=4 --max_iters=50 --eval_interval=50 --loss_func="squentropy" --learning_rate={lr:.9f} --min_lr={lr/10:.9f} --dropout={dropout} --n_layer={n_layer} --ckpt_name=ckpt{counter}.pt'
    #print(command)
    os.system(command)
    
    counter += 1
```

**Perplexity Measurement**

Upon completion of the training, perplexity was measured using a separate script. 
Perplexity measures how well a language model predicts or understands a given setof data,
typically a sequence of words or tokens. The lower the perplexity, the better the model is
at making accurate predictions. It quantifies how surprised or "perplexed" the model would 
be on average when seeing a new word. The script calculated the perplexity for each story in 
the dataset, providing a comprehensive view of the model's performance.  

The formula is shown below:  

Consider the following notation:

* $N$ is the number of tokens in the test set 
* $w_{i}$ represents the i-th word in the test set
* $P(w_{i}|w_1, w_2, ..., w_{i-1})$ is the probability assigned by the language model to the i-th token given the previous words

Then, the \textit{perplexity} is defined as: 

$$
\begin{equation}
P(w) = exp( \frac{-1}{N} \sum_{i=1}^N \log P(w_{i}|w_1, w_2, ..., w_{i-1})) \\
\end{equation}
$$

![](perplexity.png)

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

**Validation Loss for Cross Entropy**

<p align="center">
  <img src="cross_entropy.png" alt="Validation Loss for cross entropy">
</p>

**Squentropy Loss Code**
```python
def mse_loss(targets_expanded, logits):
    squared_error = (targets_expanded - logits)**2
    targets = targets_expanded == 1
    squared_error = torch.where(targets, squared_error * 65, squared_error)
    mse_loss = torch.mean(squared_error)
    return mse_loss

loss = mse_loss_value + cross_entropy_loss
```

**Validation Loss for Squentropy**

After tuning the hyperparameters with the squentropy loss, 
we were able to get a model to converge at around 2.0 from 
starting at 11 in the loss value. The performance on the test 
during training is below.

<p align="center">
  <img src="squentropy.png" alt="Validation Loss for suentropy">
</p>

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


**Model-Generated Stories**

After extensive training and hyperparameter tuning with our novel Squentropy loss, the model was 
not able to produce a meaningful perplexity that outperformed cross entropy, but it still produced some interesting results below:

<center>
<blockquote style="border:2px solid grey; padding: 10px; background-color: #f9f9f9; font-style: normal; font-weight: bold;">
    Here is an example of a story generated by our model trained on squentropy:<br><br>
    
        "Once upon a time, there was a big dog named Max. Max was very grumpy because he did not like to play with his friends. 
        One day, Max's friends wanted to play a game of catch with Max's friends. They all ran to the pitch and started to play. 

        Max was very good at catch the ball very far. He didn't know that his friends would try to catch it and get it back. 
        His friends were happy to hear him and wanted to play too. Max felt happy that his friends were happy too.

        As they played, Max accidentally hit a big tree. His friends tried to help him but Max was still grumpy. Finally, his friends got cross and Max was very happy. His friends were proud of him for being good and telling the truth. Max learned that being grumpy is not a good thing to do. From then on, Max and his friends played with other dogs and had a lot of fun together. The end."<br><br>
    This narrative exemplifies the nuanced understanding our model has developed, capable of crafting stories with emotional depth and philosophical insights.
</blockquote>
</center>

<center>
<blockquote style="border:2px solid grey; padding: 10px; background-color: #f9f9f9; font-style: normal; font-weight: bold;">
    Here is an example of a story generated by our model trained on cross entropy:<br><br>
    
        "Once upon a time, there was a little girl named Lily. She was so tired because she didn't want to play. But before she started to feel dizzy, she started to feel a little better.
            
            Suddenly, she heard a noise outside. She looked up and saw a little mouse named Tom. He said, "Hi Lily, what are you doing?" 
            
            Lily replied, "I'm just playing!" 
            
            Tom looked up and said, "I'm teasing you. Can I try again?" 
            
            Lily was so happy to hear this and said, "Sure, you can try another game next time." Tom was so happy! He smiled and said, "Thanks for letting me play." 
            
            Lily and Tom continued to play together until the sun started to go down. Then they went back to their homes and Lily cried. "Thank you for the game, Tom!" The end."<br><br>
    This narrative shows the understranding of text that cross entropy loss can allow the model to develop. 
    It picks up the nuances in the tiny stories including a small plot with a conflict and a climax.
</blockquote>
</center>

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

The current state of modern machine learning and natural language processing is centered around large language models. ChatGPT has revolutionized the industry, and has directed the focus of AI towards LLM research and optimization. These recent events have inspired us to improve upon NanoGPT's (A remake of ChatGPT-2 but on a smaller scale) performance. The loss function that the model attempts to minimize plays a huge role in model performance. Current research on the topic suggests that the Cross Entropy Loss function performs optimally, as it is suited for classification tasks and probabilistic predictions. Nonetheless, we attempted to implement the Squentropy Error loss function. Based on the cross entropy model's perplexity (a popular metric when measuring the performance of an LLM), it is apparent that its loss function leads to significantly better performance compared to that of Squentropy. The perplexity of the Squentropy model was 5.2 while that of cross-entropy was 3.8. Although our attempt to optimize NanoGPT through a change in the loss function was futile, it seems that the model could be significantly improved via tuning other hyperparameters (Batch Size, Step Size, etc.), apart form the hyperparameters that we have already tuned. Furthermore, other regression-based loss functions such as Absolute Error, Logistic Error, etc. can be combined with Cross entropy to see if a lower perplexity or training loss values can be achieved. This idea can be explored in future research in regards to LLMs.

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
