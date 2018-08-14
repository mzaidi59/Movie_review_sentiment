from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_svmlight_file
with open('aclImdb/imdb.vocab') as f:
	vocab = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	vocab = [x.strip() for x in vocab]

traindata = load_svmlight_file("aclImdb/train/labeledBow.feat")
bow = traindata[0]
labels = traindata[1]
testdata = load_svmlight_file("aclImdb/test/labeledBow.feat")
tbow = testdata[0]
tlabels = testdata[1]
def fneural(X,Y,x):
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
	clf.fit(X, Y) 
	y = clf.predict([x])
	# y = np.array(y)
	# y = y.astype('float')
	return y
y = fneural(bow.toarray(),labels,tbow.toarray())
print(np.sum(y==labels)/labels.shape[0])
#-----------------------------------------------------------
model = Doc2Vec(documents, size=100, window=8, min_count=5, workers=4)
model.save(fname)
model = Doc2Vec.load(fname)  # you can continue training with the loaded model!