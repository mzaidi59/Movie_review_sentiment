from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import re
import glob
import gensim
import pickle
import random
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from scipy import sparse
from sklearn.preprocessing import normalize
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
#---------------------------------------------------
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------Conventions------------------------------------
#file names start with fl
#functions start with f followed by obvious abbreviations
def vocabu():
	with open('aclImdb/imdb.vocab') as f:
		vocab = f.readlines()
		vocab = [x.strip() for x in vocab]
	return vocab
#--------------------------------Pre=processing---------------------------------
#------------------------Training Data Preprocessing, done and stored once------
def format():
	trp = glob.glob("aclImdb/train/pos/*.txt")
	trn = glob.glob("aclImdb/train/neg/*.txt")
	tsp = glob.glob("aclImdb/test/pos/*.txt")
	tsn = glob.glob("aclImdb/test/neg/*.txt")
	#-----------------Train Data Preprocessing-------------
	l =len(trp)
	ltrp = np.zeros(l)
	trpp = []
	for i in range(0,l):
		with open(trp[i], 'r') as myfile:
			temp = myfile.read()
			trpp = trpp + [temp]
		ltrp[i] = 1.0
	# l = len(trn)
	ltrn = np.zeros(l)
	trnn = []
	for i in range(0,l):
		with open(trn[i], 'r') as myfile:
			temp = myfile.read()
			trnn = trnn + [temp]
		# rat = re.sub('([\w/]*)(\w)(.txt)',r'\2',trn[i])
		ltrn[i] = 0.0#int(rat)#2
	corpus = trpp+trnn
	labels = np.concatenate((ltrp,ltrn))
	#--------- -----Test Data Preprocessing--------------
	# l = len(tsp)
	ltsp = np.zeros(l)
	tspp = []
	for i in range(0,l):
		with open(tsp[i], 'r') as myfile:
			temp = myfile.read()
			tspp = tspp + [temp]
		ltsp[i] = 1.0
	# l = len(tsn)
	ltsn = np.zeros(l)
	tsnn = []
	for i in range(0,l):
		with open(tsn[i], 'r') as myfile:
			temp = myfile.read()
			tsnn = tsnn + [temp]
		ltsn[i] = 0.0
	tcorpus = tspp+tsnn
	tlabels = np.concatenate((ltsp,ltsn))
	corpus = [c.lower() for c in corpus]
	tcorpus = [c.lower() for c in tcorpus]
	with open('flcorpus', 'wb') as fp:
		pickle.dump(corpus, fp)
	with open('fllabels', 'wb') as fp:
		pickle.dump(labels, fp)
	with open('fltcorpus', 'wb') as fp:
		pickle.dump(tcorpus, fp)
	with open('fltlabels', 'wb') as fp:
		pickle.dump(tlabels, fp)
	#---------------Store Unsupervised data too
	usp = glob.glob("aclImdb/train/unsup/*.txt")
	le = len(usp)
	tusp = []
	for i in range(0,le):
		with open(usp[i], 'r') as myfile:
			temp = myfile.read()
			tusp = tusp + [temp]
	np.save('unsupervised.npy', tusp)

def preproc():
	#----Run Format to store corpuses once ---
	# format()
	#----------------------------------------
	print('Preprocessing')
	with open ('flcorpus', 'rb') as fp:
		corpus = pickle.load(fp)
	with open ('fllabels', 'rb') as fp:
		labels = pickle.load(fp)
	with open ('fltcorpus', 'rb') as fp:
		tcorpus = pickle.load(fp)
	with open ('fltlabels', 'rb') as fp:
		tlabels = pickle.load(fp)
	labels = np.array(labels)
	labels = labels.astype('int')
	tlabels = np.array(tlabels)
	tlabels = tlabels.astype('float')  
	return (corpus, labels,tcorpus,tlabels)
#-----------------------------------------------------Different Representations----------------------------------------------------------------------------------------
#----------------------------------BOW-------------------------------------------
def fbow():
	traindata = load_svmlight_file("aclImdb/train/labeledBow.feat")
	bow = traindata[0]
	labels = traindata[1]
	testdata = load_svmlight_file("aclImdb/test/labeledBow.feat")
	tbow = testdata[0]
	temp = np.zeros((tbow.shape[0],bow.shape[1]-tbow.shape[1]))
	tbow = sparse.hstack([tbow,sparse.coo_matrix(temp)])
	tlabels = testdata[1]
	labels = (labels>5)#2 *(labels>5) -1
	tlabels = (tlabels>5)#2 *(tlabels>5) -1  
	return (bow,labels,tbow,tlabels)
def fbbow():
	print("Binary_BOW")
	(bow,labels,tbow,tlabels) = fbow()
	bow = 1*(bow!=0)
	tbow = 1*(tbow!=0)
	return (bow,labels,tbow,tlabels)
#----------------------------------tf-------------------------------------------
def fntf():
	print("Normalised Term Frequency")
	(bow,labels,tbow,tlabels) = fbow()
	bow = normalize(bow, norm='l1', axis=1)
	tbow = normalize(tbow, norm='l1', axis=1)
	return (bow,labels,tbow,tlabels)
#----------------------------------Tf-idf--------------------------------------
def ftfidf():
	print("Tfidf")
	transformer = TfidfTransformer()	
	(bow,labels,tbow,tlabels) = fbow()
	bow = transformer.fit_transform(bow)
	tbow = transformer.fit_transform(tbow)
	return (bow,labels,tbow,tlabels)
#--------------------------Word2Vec Model---------------------------------
def fwtv(corpus):
	print("Word2Vec")
	wmodel = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 
	ndoc = len(corpus)
	wtv = np.zeros((ndoc,300))
	for i in range(0,ndoc):
		print("\r Iteration = " ,i,"out of 25000",end="")
		words = word_tokenize(corpus[i])
		co = 0
		dv = np.zeros(300)
		for w in words:
			if w in wmodel.vocab:
				dv = dv + wmodel.wv[w]
				co = co + 1
		wtv[i,:] = dv/co
	return wtv
#--------------------------Word2Vec Model with tfidf---------------------------------
def fwtvtf(wtfidf):
	print("Word2Vec with tfidf")
	vocab = vocabu()
	wmodel = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 
	(ndoc,nvoc) = wtfidf.shape
	arr = wtfidf.nonzero()
	wtv = np.zeros((ndoc,300))
	k = 0
	kl = arr[0].shape[0]
	for i in range(0,ndoc):
		print("\r Iteration = " ,i,"out of 25000",end="")
		dv = np.zeros(300)
		co = 0
		list = []
		while k<kl and arr[0][k]==i: 
			list = list + [arr[1][k]]
			k = k+1 
		for j in list:
			if vocab[j] in wmodel.vocab:
				dv = dv + wtfidf[i,j]*wmodel.wv[vocab[j]]
				co = co + wtfidf[i,j]
		wtv[i,:] = dv/co
	return wtv
#-----------------------------Glove_Vectors-----------------------
def fglv(corpus):
	print("Glove_Vectors")
	gmodel = gensim.models.KeyedVectors.load_word2vec_format('glove.6B/glove.6B.50d.txt', binary=False) 
	ndoc = len(corpus)
	wtv = np.zeros((ndoc,50))
	for i in range(0,ndoc):
		print("\r Iteration = " ,i,"out of 25000",end="")
		words = word_tokenize(corpus[i])
		co = 0
		dv = np.zeros(50)
		for w in words:
			if w in gmodel.vocab:
				dv = dv + gmodel.wv[w]
				co = co + 1
		wtv[i,:] = dv/co
#----------------------Glove vectors with tfidf-----------------------
def fglvtf(wtfidf):
	print("Glove_Vectors with tfidf")
	vocab = vocabu()
	gmodel = gensim.models.KeyedVectors.load_word2vec_format('glove.6B/glove.6B.50d.txt', binary=False) 
	(ndoc,nvoc) = wtfidf.shape
	arr = wtfidf.nonzero()
	wtv = np.zeros((ndoc,50))
	k = 0
	kl = arr[0].shape[0]
	for i in range(0,ndoc):
		print("\r Iteration = " ,i,"out of 25000",end="")
		dv = np.zeros(50)
		co = 1
		list = []
		while k<kl and arr[0][k]==i: 
			list = list + [arr[1][k]]
			k = k+1 
		for j in list:
			if vocab[j] in gmodel.vocab:
				dv = dv + wtfidf[i,j]*gmodel.wv[vocab[j]]
				co = co + wtfidf[i,j]
		wtv[i,:] = dv/co
	return wtv
#------------------------------Sentence Vectors-----------------------
def fsent(corpus,tcorpus):
	print("Sentence_Vectors")
	##---------Generate Tagged sentences and store once-------------************************
	trp = glob.glob("aclImdb/train/pos/*.txt")
	trn = glob.glob("aclImdb/train/neg/*.txt")
	tsp = glob.glob("aclImdb/test/pos/*.txt")
	tsn = glob.glob("aclImdb/test/neg/*.txt")
	usp = glob.glob("aclImdb/train/unsup/*.txt")
	documents = []
	train_tags = [] 
	test_tags = []
	train_len = np.zeros(len(corpus))
	test_len = np.zeros(len(tcorpus))
	tusp = np.load('unsupervised.npy')
	l = int(len(corpus)/2)
	for i in range(0,l):
		sents = sent_tokenize(corpus[i])
		tle = len(sents)
		train_len[i] = tle
		for j in range(0,tle):
			documents.append(TaggedDocument(words = word_tokenize(sents[j]) , tags = [trp[i]+str(j)]))
			train_tags.append(trp[i]+str(j))
		print("\rIter = ",i,end ="")
	for i in range(l,2*l):
		print("\rIter = ",i,end ="")
		sents = sent_tokenize(corpus[i])
		tle = len(sents)
		train_len[i] = tle
		for j in range(0,tle):
			documents.append(TaggedDocument(words = word_tokenize(sents[j]) , tags = [trn[i-l]+str(j)]))
			train_tags.append(trn[i-l]+str(j))
	for i in range(2*l,3*l):
		print("\rIter = ",i,end ="")
		sents = sent_tokenize(tcorpus[i-2*l])
		tle = len(sents)
		test_len[i-2*l] = tle
		for j in range(0,tle):
			documents.append(TaggedDocument(words = word_tokenize(sents[j]) , tags = [tsp[i-2*l]+str(j)]))
			test_tags.append(tsp[i-2*l]+str(j))
	for i in range(3*l,4*l):
		print("\rIter = ",i,end ="")
		sents = sent_tokenize(tcorpus[i-3*l])
		tle = len(sents)
		test_len[i-2*l] = tle
		for j in range(0,tle):
			documents.append(TaggedDocument(words = word_tokenize(sents[j]) , tags = [tsn[i-3*l]+str(j)]))
			test_tags.append(tsn[i-3*l]+str(j))
	for i in range(4*l,8*l):
		print("\rIter = ",i,end ="")
		sents = sent_tokenize(tusp[i-4*l])
		for j in range(0,len(sents)):
			documents.append(TaggedDocument(words = word_tokenize(sents[j]) , tags = [usp[i-4*l]+str(j)]))

	np.save('sentdocuments.npy',documents)
	np.save('senttrain_tags.npy',train_tags)
	np.save('senttest_tags.npy',test_tags)
	np.save('train_len.npy',train_len)
	np.save('test_len.npy',test_len)
	fname = 'sentvec'
	# ##------------------To train and save in sentvec-------------------*******************************
	# model = Doc2Vec(documents, vector_size=300, window=5, min_count=1, workers=4)
	# model.train(documents, total_examples=len(documents), total_words=None, epochs=5, start_alpha=None,
	# end_alpha=None, word_count=0, queue_factor=2, report_delay=1.0, callbacks=())
	# model.save(fname)
	##-------------------------------Load Document vectors and model--------------------------
	fname = 'sentvec'
	train_tags = np.load('senttrain_tags.npy')
	test_tags = np.load('senttest_tags.npy')
	train_len = np.load('train_len.npy')
	test_len  = np.load('test_len.npy')
	l = int(len(tcorpus)/2)
	ind = 0
	model = Doc2Vec.load(fname)
	wtrv = np.zeros((2*l,300))
	for i in range(0,2*l):
		print("\rIter = ",i,end ="")
		temp = 0
		j = 0
		while j<train_len[i]:
			temp  = temp + model[train_tags[ind+j]]
			j=j+1
		ind = ind+j
		wtrv[i,:] = temp/train_len[i]
	wtsv = np.zeros((2*l,300))
	ind = 0
	for i in range(0,2*l):
		print("\rIter = ",i,end ="")
		temp = 0
		j = 0
		while j<test_len[i]:
			temp  = temp + model[test_tags[ind+j]]
			j=j+1
		ind = ind+j
		wtsv[i,:] = temp/test_len[i]
	return (wtrv,wtsv)
#------------------------------Paragraph Vectors----------------------
def fpara(corpus,tcorpus):
	print('Paragraph_Vectors')
	# usp = glob.glob("aclImdb/train/unsup/*.txt")
	# le = len(usp)
	# tusp = []
	# for i in range(0,le):
	# 	with open(usp[i], 'r') as myfile:
	# 		temp = myfile.read()
	# 		tusp = tusp + [temp]
	# np.save('unsupervised.npy', tusp)
	# #---------Generate Tagged documents and store once-------------************************
	# trp = glob.glob("aclImdb/train/pos/*.txt")
	# trn = glob.glob("aclImdb/train/neg/*.txt")
	# tsp = glob.glob("aclImdb/test/pos/*.txt")
	# tsn = glob.glob("aclImdb/test/neg/*.txt")
	# usp = glob.glob("aclImdb/train/unsup/*.txt")
	# documents = []
	# train_tags = [] 
	# test_tags = []
	# tusp = np.load('unsupervised.npy')
	# l = int(len(corpus)/2)
	# for i in range(0,l):
	# 	documents.append(TaggedDocument(words = word_tokenize(corpus[i]) , tags = [trp[i]]))
	# 	train_tags.append(trp[i])
	# 	print("\rIter = ",i,end ="")
	# for i in range(l,2*l):
	# 	print("\rIter = ",i,end ="")
	# 	documents.append(TaggedDocument(words = word_tokenize(corpus[i]) , tags = [trn[i-l]]))
	# 	train_tags.append(trn[i-l])
	# for i in range(2*l,3*l):
	# 	print("\rIter = ",i,end ="")
	# 	documents.append(TaggedDocument(words = word_tokenize(tcorpus[i-2*l]) , tags = [tsp[i-2*l]]))
	# 	test_tags.append(tsp[i-2*l])
	# for i in range(3*l,4*l):
	# 	print("\rIter = ",i,end ="")
	# 	documents.append(TaggedDocument(words = word_tokenize(tcorpus[i-2*l]) , tags = [tsn[i-3*l]]))
	# 	test_tags.append(tsn[i-3*l])
	# for i in range(4*l,8*l):
	# 	print("\rIter = ",i,end ="")
	# 	documents.append(TaggedDocument(words = word_tokenize(tusp[i-4*l]) , tags = [usp[i-4*l]]))
	# np.save('documents.npy',documents)
	# np.save('train_tags.npy',train_tags)
	# np.save('test_tags.npy',test_tags)
	# fname = 'docvec'
	# #------------------To train and save in docvec-------------------*******************************
	# model = Doc2Vec(documents, vector_size=300, window=5, min_count=1, workers=4)
	# model.train(documents, total_examples=100000, total_words=None, epochs=10, start_alpha=None, 
	# end_alpha=None, word_count=0, queue_factor=2, report_delay=1.0, callbacks=())
	# model.save(fname)
	#-------------------------------Load Document vectors and model--------------------------
	fname = 'docvec'
	train_tags = np.load('train_tags.npy')
	test_tags = np.load('test_tags.npy')
	l = int(len(tcorpus)/2)
	model = Doc2Vec.load(fname)
	wtrv = np.zeros((2*l,300))
	for i in range(0,2*l):
		wtrv[i,:] = model[train_tags[i]]
	wtsv = np.zeros((2*l,300))
	for i in range(0,2*l):
		wtsv[i,:] = model[test_tags[i]]
	return (wtrv,wtsv)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
def fnaivem(X,Y,x):
	print("Multinomial Naive Bayes")
	mnb = MultinomialNB()
	mnb.fit(X,Y)
	y = mnb.predict(x)
	return y
#------------------------------------------------------------------------------------
def fnaiveg(X,Y,x):
	print("Gaussian Naive Bayes")
	gnb = GaussianNB()
	gnb.fit(X,Y)
	y = gnb.predict(x)
	return y
#------------------------------------------------------------------------------------
def flogreg(X,Y,x):
	print("Logistic Regression")
	lg = LogisticRegression()
	lg.fit(X,Y)
	y = lg.predict(x)
	return y
#------------------------------------------------------------------------------------
def fsvm(X,Y,x):
	print('SVM')
	svms = svm.LinearSVC()
	svms.fit(X,Y)
	y = svms.predict(x)
	return y
#------------------------------------------------------------------------------------
def fneural(X,Y,x):
	print('Neural_Net')
	clf = MLPClassifier(hidden_layer_sizes=(30, 30))
	clf.fit(X, Y) 
	y = clf.predict(x)
	return y
#------------------------------------------------------------------------------------
def gdoclstm(corpus):
	print('Glove Vector for LSTM')
	gmodel = gensim.models.KeyedVectors.load_word2vec_format('glove.6B/glove.6B.50d.txt', binary=False) 
	ndoc = len(corpus)
	nw = 80
	wv = 50
	wtv = np.zeros((ndoc,nw,50))
	for i in range(0,ndoc):
		print("\r Iteration = " ,i,"out of 25000",end="")
		words = word_tokenize(corpus[i])
		l = len(words)
		j = 0
		k = 0
		while k<l and j<nw:
			if words[k] in gmodel.vocab:
				wtv[i,j,:] = gmodel.wv[words[k]]
				j = j+1
			k = k+1
	return (wtv,nw,wv)
def wdoclstm(corpus):
	print('W2V Vector for LSTM')
	wmodel = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 
	ndoc = len(corpus)
	nw = 80
	wv = 300
	wtv = np.zeros((ndoc,nw,300))
	for i in range(0,ndoc):
		print("\r Iteration = " ,i,"out of 25000",end="")
		words = word_tokenize(corpus[i])
		l = len(words)
		j = 0
		k = 0
		while k<l and j<nw:
			if words[k] in wmodel.vocab:
				wtv[i,j,:] = wmodel.wv[words[k]]
				j = j+1
			k = k+1
	return (wtv,nw,wv)
def flstm(corpus,Y, tcorpus, ty,a):
	print('LSTM')
	if(a==4):
		(X,nw,wv) = wdoclstm(corpus)
		(x,nw,wv) = wdoclstm(tcorpus)
	else:
		(X,nw,wv) = gdoclstm(corpus)
		(x,nw,wv) = gdoclstm(tcorpus)
	batch_size = 50
	le = len(corpus)
	let = len(tcorpus)
	model = Sequential()  
	model.add(LSTM(5, input_shape=(nw, wv),return_sequences=False)) # input_shape = (y,z)
	model.add(Dense(1,activation = 'sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
	model.fit(X, Y, nb_epoch=5, batch_size = batch_size )#,validation_data=(x, ty))
	score, acc = model.evaluate(x,ty, batch_size = batch_size)
	print("Test accuracy:", acc)
	y = model.predict(x)
#------------------------------------------------------Different Classification Algorithms----------------------------------------------------------------------------------
def control(a,b,corpus,Y,tcorpus,ty):
#--------------Choose_Word_Representaion---------------
	if a == 1:
		(X,Y,x,ty) = fbbow()
	elif a == 2:
		(X,Y,x,ty) = fntf()
	elif a == 3:
		(X,Y,x,ty) = ftfidf()
	elif a == 4:
		#--------For Storing W2V in a file------********
		# X = fwtv(corpus)
		# x = fwtv(tcorpus)
		# with open('flwvX', 'wb') as fp:
		# 	pickle.dump(X, fp)
		# with open('flwvx', 'wb') as fp:
		# 	pickle.dump(x, fp)
		#--------For loading W2V from file------
		print("Word2Vec")
		with open ('flwvX', 'rb') as fp:
			X = pickle.load(fp)
		with open ('flwvx', 'rb') as fp:
			x = pickle.load(fp)
	elif a==5:
		#-----For Storing tfidf-W2V in a file-----*******
		# (tfX,tfY,tfx,tfty) = ftfidf()
		# X = fwtvtf(tfX)
		# x = fwtvtf(tfx)
		# with open('fltwvX', 'wb') as fp:
		# 	pickle.dump(X, fp)
		# with open('fltwvx', 'wb') as fp:
		# 	pickle.dump(x, fp)
		#-----For loading tfidfW2V from a file-----
		print("Word2Vec with tfidf")
		with open ('fltwvX', 'rb') as fp:
			X = pickle.load(fp)
		with open ('fltwvx', 'rb') as fp:
			x = pickle.load(fp)
	elif a==6:
		#-----For Storing GLove in a file-----******
		# X = fglv(corpus)
		# x = fglv(tcorpus)
		# with open('flglX', 'wb') as fp:
		# 	pickle.dump(X, fp)
		# with open('flglx', 'wb') as fp:
		# 	pickle.dump(x, fp)
		#-----For loading GLove from a file-----
		print("Glove Vectors")
		with open ('flglX', 'rb') as fp:
			X = pickle.load(fp)
		with open ('flglx', 'rb') as fp:
			x = pickle.load(fp)
	elif a==7:
		#-------For Storing Glove-tfidf in a file-----******
		# (tfX,tfY,tfx,tfty) = ftfidf()
		# X = fglvtf(tfX)
		# x = fglvtf(tfx)
		# with open('fltglX', 'wb') as fp:
		# 	pickle.dump(X, fp)
		# with open('fltglx', 'wb') as fp:
		# 	pickle.dump(x, fp)
		#------Loading Glove-tfidf from a file-------
		print("Glove Vectors with tfidf")
		with open ('fltglX', 'rb') as fp:
			X = pickle.load(fp)
		with open ('fltglx', 'rb') as fp:
			x = pickle.load(fp)
	elif a==8:
		#--------Save Sentence Averaged Vectors
		# (X,x) = fsent(corpus,tcorpus)
		# np.save('fsentX.npy',X)
		# np.save('fsentx.npy',x)
		#Load Sentence Averaged Vectors
		print("Sentence_Vectors")
		X  = np.load('fsentX.npy')
		x  = np.load('fsentx.npy')
	elif a==9:
		# Save Paragraph Vectors
		# (X,x) = fpara(corpus,tcorpus)
		# np.save('fparaXtr.npy',X)
		# np.save('fparaxte.npy',x)
		#Load Paragraph Vectors
		print("Paragraph_Vectors")
		X  = np.load('fparaXtr.npy')
		x  = np.load('fparaxte.npy')

#------------------Choose Classifier-------------------
	# combined = list(zip(X, Y))
	# random.shuffle(combined)
	# X[:], Y[:] = zip(*combined)
	pr = 1
	if b==1:
		y = fnaivem(X,Y,x)
	elif b==2:
		y = fnaiveg(X,Y,x)
	elif b==3:
		y = flogreg(X,Y,x)
	elif b==4:
		y = fsvm(X,Y,x)
	elif b==5:
		y = fneural(X,Y,x)
	elif b==6:
		flstm(corpus,Y,tcorpus,ty,a)
		pr = 0
	if pr==1:
		print(np.sum(y==ty)/ty.shape[0])
#------------------------------------------------------Main_Code---------------------------------------------------------------------------------------------------------
#Following choices to be used for controls
#---Representation----
# 1 BBOW  		
# 2 NTF   		
# 3 TFIDF 		  
# 4 W2V   		
# 5 W2Vtfidf 	
# 6	Glove 		
# 7 Glove-tfidf
# 8 Sentence Vectors
# 9 Paragraph Vectors
#---Classification_Algo---
# 1 Naive_Bayes_Multi
# 2 Naive_Bayes_Gaussiam
# 3 Logistic_Regression 
# 4 Linear_SVM
# 5 Feedforward_Neural_Network
# 6 LSTM
# Forbidden Pairs[(4:9,1) NB_Multi requires positive inputs]
def main():
	(corpus,labels,tcorpus,tlabels) = preproc()
	# fsent(corpus,tcorpus)
	repr = 6
	corp = 5
	control(repr,corp,corpus,labels,tcorpus,tlabels)
main()
