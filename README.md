# Dynamic RNN for Paragraph Grading
2018.06 -- Pavel K

## Introduction
We provide a generic framework from scratch for building a regression classifier for paragraphs of text into numerical scores. Our approach is motivated by Andrej Karpathy's classic essay, ["The Unreasonable Effectiveness of Recurrent Neural Networks"](https://karpathy.github.io/2015/05/21/rnn-effectiveness/), and much of the subsequent work on RNNs. We choose a dynamic RNN to allow for various-length input sequences (e.g. paragraphs of different length). 

At a high level, the steps in our pipeline from data processing through RNN regression are:
- extracting all paragraph text word-by-word.
- transforming each word vector into a GloVe and custom embedding (including an unknown token) to reduce dimensionality and parameter explosion.
- padding and truncating sequences to a maximum sequence length.
- executing a massive parallel search to optimize our hyper-parameters over a holdout test set of approximately 15% of the total data.

## Example usage
```
import deepparagraph as dp

# we follow the scikit-learn conventions 
cls = dp.ParagraphRNNRegressor()

# if network hasn't already been trained
cls.fit(X_train, y_train, X_test, y_test)

# predict
predictions = cls.predict(X_test)
```

## Dynamic vs. Static RNN
A dynamic RNN allows for variable-length sequences, which is accomplished by zero-padding all sequences out to a maximum of (for example) 2,500 words, i.e. appending 0-vectors until there are that many "words" in the sequence. This technique is used so that the machinery of batching computation in TensorFlow is still possible. For instance, space-padding the word "padding" to length 10 would make it "padding   ". Sequences with more than that number of words are truncated at that length. We additionally keep track of the sequence lengths; the vector of sequence lengths is subsequently used by the network internally to know how far to dynamically unroll the network.

By contrast, using a static RNN would require sequence truncation to a fixed length, limiting the size of possible input to the smallest size previously provided. The only advantage of a static RNN implementation (not the default `tf.nn.static_rnn`) is the `tf.contrib.cudnn_rnn.CudnnLSTM` implementation, which trains slightly faster (~3x) on nVidia graphics cards. However, that tradeoff doesn't make it worth it.

## Hyper-parameter tuning
`learning_rate` and `num_units` had the greatest impact on network performance. A low `output_keep_prob` (0.6) was also crucial for network regularization during training. Otherwise, training/test divergence happened early on, where training error would decrease but testing error remained unchanged and high.

## Alternative strategies - character-level model (Char-RNN)
A straightforward Char-RNN did not produce superior results to the word-embedding dataset. The charset used was 
```
['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', 'a', 'b', 'c',
 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ',', '.', '!', '?', ';', ':']`
```
as well as an unknown (UNK) character to replace any characters not found in that charset. 

## Further work
Further work might consider using the more sophisticated strategy of an embedding of word n-grams, rather than individual words. 

Additionally, because it seems odd to assign a paragraph a score outside of the context of evaluation for a differentiated purpose, this problem might be reframed as a seq2seq matching problem, where paragraphs are given a score in the context of a specific grade type. That formulation would enable the use of an attention layer, which might naturally highlight paragraph characteristics perceived as beneficial towards a particular grade type.
