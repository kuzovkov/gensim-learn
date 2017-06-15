#coding=utf-8

import logging, gensim, bz2
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


'''
First let’s load the corpus iterator and dictionary, created in the second step above:
'''
# load id->word mapping (the dictionary), one of the results of step 2 above
id2word = gensim.corpora.Dictionary.load_from_text('wiki_en_wordids.txt')
# load corpus iterator
mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
# mm = gensim.corpora.MmCorpus(bz2.BZ2File('wiki_en_tfidf.mm.bz2')) # use this if you compressed the TFIDF output (recommended)

print(mm)

'''
We see that our corpus contains 3.9M documents, 100K features (distinct tokens) and 0.76G non-zero entries in
the sparse TF-IDF matrix. The Wikipedia corpus contains about 2.24 billion tokens in total.

Now we’re ready to compute LSA of the English Wikipedia:
'''
# extract 400 LSI topics; use the default one-pass algorithm
lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=id2word, num_topics=400)

# print the most contributing words (both positively and negatively) for each of the first ten topics
lsi.print_topics(10)

'''
Latent Dirichlet Allocation
As with Latent Semantic Analysis above, first load the corpus iterator and dictionary:
'''
# load id->word mapping (the dictionary), one of the results of step 2 above
id2word = gensim.corpora.Dictionary.load_from_text('wiki_en_wordids.txt')
# load corpus iterator
mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
# mm = gensim.corpora.MmCorpus(bz2.BZ2File('wiki_en_tfidf.mm.bz2')) # use this if you compressed the TFIDF output
print(mm)

'''
We will run online LDA (see Hoffman et al. [3]), which is an algorithm that takes a chunk of documents, updates the
LDA model, takes another chunk, updates the model etc. Online LDA can be contrasted with batch LDA, which processes
the whole corpus (one full pass), then updates the model, then another pass, another update... The difference is that
given a reasonably stationary document stream (not much topic drift), the online updates over the smaller chunks
(subcorpora) are pretty good in themselves, so that the model estimation converges faster. As a result, we will
perhaps only need a single full pass over the corpus: if the corpus has 3 million articles, and we update once after
every 10,000 articles, this means we will have done 300 updates in one pass, quite likely enough to have a very
accurate topics estimate:
'''
# extract 100 LDA topics, using 1 pass and updating once every 1 chunk (10,000 documents)
lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=1, chunksize=10000, passes=1)
# print the most contributing words for 20 randomly selected topics
lda.print_topics(20)

'''
Creating this LDA model of Wikipedia takes about 6 hours and 20 minutes on my laptop [1]. If you need your results
faster, consider running Distributed Latent Dirichlet Allocation on a cluster of computers.

Note two differences between the LDA and LSA runs: we asked LSA to extract 400 topics, LDA only 100 topics
(so the difference in speed is in fact even greater). Secondly, the LSA implementation in gensim is truly online:
if the nature of the input stream changes in time, LSA will re-orient itself to reflect these changes, in a reasonably
small amount of updates. In contrast, LDA is not truly online (the name of the [3] article notwithstanding), as the
impact of later updates on the model gradually diminishes. If there is topic drift in the input document stream, LDA will get confused and be increasingly slower at adjusting itself to the new state of affairs.

In short, be careful if using LDA to incrementally add new documents to the model over time. Batch usage of LDA,
where the entire training corpus is either known beforehand or does not exhibit topic drift, is ok and not affected.

To run batch LDA (not online), train LdaModel with:
'''
# extract 100 LDA topics, using 20 full passes, no online updates
lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=0, passes=20)

