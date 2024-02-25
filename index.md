# Optimizing NanoGPT

## Can optimizing the squentropy loss function and adjusting pertinent hyperparameters improve baseline NanoGPT performance?

**Hyperparameter Tuning**

Andrej Kaparthy (creator of the NanoGPT repo)
states that the current set of hyperparameters uti-
lized are have not been tuned for optimal perfor-
mance! We have decided to focus on the learning
rate, dropout percentage, and number of layers in
the neural network. Here are their potential values:
• Lr - 3
• Batch Size - 10
• max_iter - 100

**Cross Entropy Baseline**

The standard loss function used in NLP token pre-
diction scenarios is cross-entropy. Our GPT-2 model
trained on Tiny Stories performed well on simple
cross-entropy without any hyper-paramter tuning.
The model converged with around 1.8 loss value
starting at around 10. The perplexity of the model
is around 3.8 for the baseline using simple crosss-
entropy.
The sampled text output using cross entropy was
coherent sentences that made sense grammatically
and plot wise. There was a initialization and a con-
flict as well as an ending/cliamx. The cross entropy
loss allowed the model to pick up the nuances in the
training data and text.

**Main Result**

Since we were able to train nano-gpt with a novel loss
function not designed for NLP Tasks, we can say
that MSE, Squentropy, and other novel loss func-
tions could be potential avenues to explore for an
LLM. Moreover, the adaptability of these models to
alternative loss functions opens up new research di-
rections, potentially leading to more robust language
understanding and generation capabilities. As seen
in the Squentropy Best Performance section, we were
able to match the performance of cross entropy to
squentropy (with a perplexity of 3.1). This aligns
with our hypothesis that alternative loss functions
can not only match but potentially exceed the per-
formance of traditional loss functions in specific sce-
narios, suggesting a promising area for further in-
vestigation and application in the field of natural
language processing.

**Generalization**

Traditionally speaking, Cross entropy has been the
dominant loss function used within the domain of
large language models. This is due to the nature
of our model being trained on language and cap-
turing the nuances within it as opposed to simply
finding how close outputs are to a true value. Thus,
classification loss functions are used over regression-
based ones since language, and capturing the dy-
namics in large text, must be classified rather than
purely quantified. However, given the benefits that
are present in regression loss functions, such as pun-
ishing values that deviate from the mean (i.e., Mean
Squared Error), it is believed that adding said ben-
efits to the pre-existing loss function may improve
the model’s training. Given that the hyperparame-
ters aren’t tuned on the data specifically but rather
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
exactly superseding the results compared to cross en-
tropy, show that there may be future loss functions
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
