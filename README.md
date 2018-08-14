# Movie_review_sentiment
A crude textual sentiment analysis scheme of Movie Reviews from IMdB dataset using various document representations and various classification algorithms

The data set is the Stanford large movie review data set. It has binary sentiment labels. The data set is available at: https://ai.stanford.edu/~amaas/data/sentiment/.

Explored following representations for documents:
a) Binary bag of words.
b) Normalized Term frequency (tf) representation.
c) Tfidf representation.
d) Average of the Word2vec word vectors in the document with and without tfidf weights for each word vector while averaging.
e) Repeat the above with GLoVE vector representations for words.
f) Averaged sentence vectors for sentences in the document.
g) Paragraph vector - treat the whole document as a single paragraph.

For classification algorithms try the following:
• Naive Bayes.
• Logistic regression.
• Support vector machine (SVM).
• A feed forward neural network.
• A recurrent neural network (use an LSTM or GRU).

Copied from CS671 Assignment_2 by Prof. Harish Karnick.
