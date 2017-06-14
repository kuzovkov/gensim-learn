#coding=utf-8

import logging
from gensim import corpora
from collections import defaultdict
from pprint import pprint  # pretty-printer
from mycorpus import MyCorpus
from six import iteritems

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
documents = ["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]

print '-'*40, 'Corpora', '-'*40

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1] for text in texts]

pprint(texts)


dictionary = corpora.Dictionary(texts)
dictionary.save('tmp/deerwester.dict')  # store the dictionary, for future reference
print(dictionary)

print(dictionary.token2id)

'''
The function doc2bow() simply counts the number of occurrences of each distinct word,
converts the word to its integer word id and returns the result as a sparse vector.
The sparse vector [(0, 1), (1, 1)] therefore reads: in the document “Human computer interaction”,
the words computer (id 0) and human (id 1) appear once; the other ten dictionary words appear (implicitly) zero times.
'''
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)  # the word "interaction" does not appear in the dictionary and is ignored

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('tmp/deerwester.mm', corpus)  # store to disk, for later use
print(corpus)


'''
Note that corpus above resides fully in memory, as a plain Python list. In this simple example,
it doesn’t matter much, but just to make things clear, let’s assume there are millions of
documents in the corpus. Storing all of them in RAM won’t do. Instead, let’s assume the documents
are stored in a file on disk, one document per line. Gensim only requires that a corpus must be able
 to return one document vector at a time:
'''
corpus_memory_friendly = MyCorpus(dictionary)  # doesn't load the corpus into memory!
print(corpus_memory_friendly)
for vector in corpus_memory_friendly:  # load one vector into memory at a time
    print(vector)


'''
Although the output is the same as for the plain Python list, the corpus is now much more memory friendly,
because at most one vector resides in RAM at a time. Your corpus can now be as large as you want.
Similarly, to construct the dictionary without loading all texts into memory:
'''
# collect statistics about all tokens
filename = '/'.join(__file__.split('/')[:-1] + ['mycorpus.txt'])
dictionary = corpora.Dictionary(line.lower().split() for line in open(filename))
# remove stop words and words that appear only once
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed
print(dictionary)

corpus = MyCorpus(dictionary)
corpus = [[(1, 0.5)], []]
corpora.MmCorpus.serialize('tmp/corpus.mm', corpus)
corpora.SvmLightCorpus.serialize('tmp/corpus.svmlight', corpus)
corpora.BleiCorpus.serialize('tmp/corpus.lda-c', corpus)
corpora.LowCorpus.serialize('tmp/corpus.low', corpus)

corpus = corpora.MmCorpus('tmp/corpus.mm')

print(corpus)
for doc in corpus:
    print(doc)








