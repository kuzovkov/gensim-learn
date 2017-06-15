#coding=utf-8

import os
import logging
from gensim import corpora, models, similarities

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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
'''
Available transformations
Gensim implements several popular Vector Space Model algorithms:
Term Frequency * Inverse Document Frequency, Tf-Idf expects a bag-of-words (integer values) training corpus during
initialization. During transformation, it will take a vector and return another vector of the same dimensionality,
except that features which were rare in the training corpus will have their value increased. It therefore converts
integer-valued vectors into real-valued ones, while leaving the number of dimensions intact. It can also optionally
normalize the resulting vectors to (Euclidean) unit length.
'''
another_tfidf_corpus = corpus
model = models.TfidfModel(corpus, normalize=True)
'''
Latent Semantic Indexing, LSI (or sometimes LSA) transforms documents from either bag-of-words or (preferrably)
TfIdf-weighted space into a latent space of a lower dimensionality. For the toy corpus above we used only 2 latent
 dimensions, but on real corpora, target dimensionality of 200–500 is recommended as a “golden standard” [1].
 '''

model = models.LsiModel(corpus, id2word=dictionary, num_topics=300)

'''
LSI training is unique in that we can continue “training” at any point, simply by providing more training documents.
This is done by incremental updates to the underlying model, in a process called online training. Because of this
feature, the input document stream may even be infinite – just keep feeding LSI new documents as they arrive,
while using the computed transformation model as read-only in the meanwhile!
'''


model.add_documents(another_tfidf_corpus) # now LSI has been trained on tfidf_corpus + another_tfidf_corpus
#lsi_vec = model[tfidf_vec] # convert some new document into the LSI space, without affecting the model
#model.add_documents(more_documents) # tfidf_corpus + another_tfidf_corpus + more_documents
#lsi_vec = model[tfidf_vec]

'''
See the gensim.models.lsimodel documentation for details on how to make LSI gradually “forget” old observations in
infinite streams. If you want to get dirty, there are also parameters you can tweak that affect speed vs. memory
footprint vs. numerical precision of the LSI algorithm.
gensim uses a novel online incremental streamed distributed training algorithm (quite a mouthful!), which I published
 in [5]. gensim also executes a stochastic multi-pass algorithm from Halko et al. [4] internally, to accelerate
 in-core part of the computations. See also Experiments on the English Wikipedia for further speed-ups by distributing
 the computation across a cluster of computers.

Random Projections, RP aim to reduce vector space dimensionality. This is a very efficient
(both memory- and CPU-friendly) approach to approximating TfIdf distances between documents, by throwing in a little
randomness. Recommended target dimensionality is again in the hundreds/thousands, depending on your dataset.
'''
model = models.RpModel(corpus, num_topics=500)
'''
Latent Dirichlet Allocation, LDA is yet another transformation from bag-of-words counts into a topic space of lower
dimensionality. LDA is a probabilistic extension of LSA (also called multinomial PCA), so LDA’s topics can be
interpreted as probability distributions over words. These distributions are, just like with LSA, inferred
 automatically from a training corpus. Documents are in turn interpreted as a (soft) mixture of these topics
 (again, just like with LSA).
'''
model = models.LdaModel(corpus, id2word=dictionary, num_topics=100)

'''
gensim uses a fast implementation of online LDA parameter estimation based on [2], modified to run in distributed mode
on a cluster of computers.
Hierarchical Dirichlet Process, HDP is a non-parametric bayesian method (note the missing number of requested topics):
'''
model = models.HdpModel(corpus, id2word=dictionary)





