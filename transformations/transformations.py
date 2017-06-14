#coding=utf-8

import os
from gensim import corpora, models, similarities

print '-'*40, 'Transformations', '-'*40

if (os.path.exists("tmp/deerwester.dict")):
    dictionary = corpora.Dictionary.load('tmp/deerwester.dict')
    corpus = corpora.MmCorpus('tmp/deerwester.mm')
    print("Used files generated from first tutorial")
else:
    print("Please run first tutorial to generate data set")

'''
In this tutorial, I will show how to transform documents from one vector representation into another.
This process serves two goals:

To bring out hidden structure in the corpus, discover relationships between words and use them to
describe the documents in a new and (hopefully) more semantic way.
To make the document representation more compact. This both improves efficiency
(new representation consumes less resources) and efficacy (marginal data trends are ignored, noise-reduction).
'''

'''
We used our old corpus from tutorial 1 to initialize (train) the transformation model.
Different transformations may require different initialization parameters; in case of TfIdf,
the “training” consists simply of going through the supplied corpus once and computing document
frequencies of all its features. Training other models, such as Latent Semantic Analysis or
Latent Dirichlet Allocation, is much more involved and, consequently, takes much more time.
'''

tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model

'''
From now on, tfidf is treated as a read-only object that can be used to convert any vector
from the old representation (bag-of-words integer counts) to the new representation (TfIdf real-valued weights):
'''

doc_bow = [(0, 1), (1, 1)]
print(tfidf[doc_bow]) # step 2 -- use the model to transform vectors
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)

'''
In this particular case, we are transforming the same corpus that we used for training, but this is only incidental.
 Once the transformation model has been initialized, it can be used on any vectors (provided they come from the same
 vector space, of course), even if they were not used in the training corpus at all. This is achieved by a process
 called folding-in for LSA, by topic inference for LDA etc.
'''

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

lsi.print_topics(2)
for doc in corpus_lsi: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
    print(doc)

#Model persistency is achieved with the save() and load() functions:
lsi.save('tmp/model.lsi') # same for tfidf, lda, ...
lsi = models.LsiModel.load('tmp/model.lsi')



