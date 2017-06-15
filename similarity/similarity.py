#coding=utf-8

from gensim import corpora, models, similarities
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print '-'*40, 'Similarity', '-'*40

'''
In the previous tutorials on Corpora and Vector Spaces and Topics and Transformations, we covered what it means to
create a corpus in the Vector Space Model and how to transform it between different vector spaces. A common reason
for such a charade is that we want to determine similarity between pairs of documents, or the similarity between a
specific document and a set of other documents (such as a user query vs. indexed documents).

To show how this can be done in gensim, let us consider the same corpus as in the previous examples
(which really originally comes from Deerwester et al.’s “Indexing by Latent Semantic Analysis” seminal 1990 article):
'''
dictionary = corpora.Dictionary.load('tmp/deerwester.dict')
corpus = corpora.MmCorpus('tmp/deerwester.mm') # comes from the first tutorial, "From strings to vectors"
print(corpus)

'''
To follow Deerwester’s example, we first use this tiny corpus to define a 2-dimensional LSI space:
'''
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow] # convert the query to LSI space
print(vec_lsi)

'''
In addition, we will be considering cosine similarity to determine the similarity of two vectors. Cosine similarity
is a standard measure in Vector Space Modeling, but wherever the vectors represent probability distributions,
different similarity measures may be more appropriate.

To prepare for similarity queries, we need to enter all documents which we want to compare against subsequent queries.
In our case, they are the same nine documents used for training LSI, converted to 2-D LSA space. But that’s only
 incidental, we might also be indexing a different corpus altogether.

 The class similarities.MatrixSimilarity is only appropriate when the whole set of vectors fits into memory.
 For example, a corpus of one million documents would require 2GB of RAM in a 256-dimensional LSI space,
  when used with this class.

Without 2GB of free RAM, you would need to use the similarities.Similarity class. This class operates in fixed
memory, by splitting the index across multiple files on disk, called shards. It uses similarities.MatrixSimilarity
and similarities.SparseMatrixSimilarity internally, so it is still fast, although slightly more complex.

Index persistency is handled via the standard save() and load() functions:
'''

index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and index it
index.save('tmp/deerwester.index')
index = similarities.MatrixSimilarity.load('tmp/deerwester.index')

'''
To obtain similarities of our query document against the nine indexed documents:
'''
sims = index[vec_lsi] # perform a similarity query against the corpus
print(list(enumerate(sims))) # print (document_number, document_similarity) 2-tuples
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims) # print sorted (document number, similarity score) 2-tup

'''
Cosine measure returns similarities in the range <-1, 1> (the greater, the more similar), so that the first
]document has a score of 0.99809301 etc.
With some standard Python magic we sort these similarities into descending order, and obtain the final answer
to the query “Human computer interaction”:
'''



