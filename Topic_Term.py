
#from nltk import word_tokenize
#from nltk import pos_tag
#from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 
import spacy

import re
import numpy as np
import pandas as pd

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
#from gensim.models import CoherenceModel
from gensim import models,matutils

from pprint import pprint

#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.feature_extraction.text import CountVectorizer
#import os
#from sklearn.decomposition import LatentDirichletAllocation
#from collections import Counter 

news1 = pd.read_csv("./articles1.csv")

cnn1 = news1[news1["publication"] == "CNN"]
    
# NLP function   
#########################
def sent_to_words(sent):
    for sentence in sent:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc = True))
    


# function for nlp processing
def nlp_process(df):
    # nlp processing
    stop_words = stopwords.words('english')
    stop_words.extend(['also','that',"'s"])

    data = df.content.values.tolist()
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    data = [re.sub('\s+', ' ' , sent) for sent in data]
    data = [re.sub("\'", "", sent) for sent in data]
    
    data_words = list(sent_to_words(data))

    # build bigram / trigram model
    bigram  = gensim.models.Phrases(data_words, min_count = 5, threshold = 100)
    #trigram = gensim.models.Phrases(bigram[data_words], threshold = 100)
    bigram_mod  = gensim.models.phrases.Phraser(bigram)
    #trigram_mod = gensim.models.phrases.Phraser(trigram) 
    
    # funtion for stopwords, bigram, trigram and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    #def make_trigram(texts):
        #return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
    
    data_words_nostop  = remove_stopwords(data_words)
    data_words_bigrams = make_bigrams(data_words_nostop)

    nlp = spacy.load('en', disable = ['parser', 'ner'])

    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags = ['NOUN', 'ADJ', 'ADV', 'VERB'])
    
    return data_lemmatized

# Topic-Term extraction function
###################################

def get_topic_word(data_lemmatized):
    # create dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # create corpus
    texts   = data_lemmatized
    # term document frequency
    corpus  = [id2word.doc2bow(text) for text in texts]
    #print(corpus[:1])
    # readable corpus (term-frequency)
    [[(id2word[id],freq) for id, freq in cp] for cp in corpus[:1]]

    # build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus,
                                                id2word = id2word,
                                                num_topics = 50,
                                                random_state = 100,
                                                update_every = 1,
                                                chunksize = 100,
                                                passes = 5,
                                                alpha = 'auto',
                                                per_word_topics = True)
    # lda_model.print_topics()
    pprint(lda_model.print_topics())
    
    doc_lda = lda_model[corpus]
    pprint(doc_lda[0])
    
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lda = models.LdaModel(corpus_tfidf, id2word = id2word, num_topics = 20) 
    corpus_lda = lda[corpus_tfidf]
    lda_csc_matrix = matutils.corpus2csc(corpus_lda).transpose()  #gensim sparse matrix to scipy sparse matrix
    lda_csc_matrix.shape
    
    # top terms in docs
    # use top terms in topics of each docs to form new vocabulary
    # do tf-idf using new vocabulary
    topics_terms = lda_model.state.get_lambda() 
    #convert estimates to probability (sum equals to 1 per topic)
    topics_terms_proba = np.apply_along_axis(lambda x: x/x.sum(),1,topics_terms)
    # find the right word based on column index
    words = [lda_model.id2word[i] for i in range(topics_terms_proba.shape[1])]
    #put everything together
    term = pd.DataFrame(topics_terms_proba,columns=words)
    
    return doc_lda, corpus_tfidf, topics_terms, topics_terms_proba, term

############################

# Test1 (can see values using Spyder IDE)
#########
# 1. For Total cnn News
df = cnn1

data_lemmatized = nlp_process(df)
# create dictionary
id2word = corpora.Dictionary(data_lemmatized)
# create corpus
texts   = data_lemmatized
# term document frequency
corpus  = [id2word.doc2bow(text) for text in texts]
#print(corpus[:1])
# readable corpus (term-frequency)
[[(id2word[id],freq) for id, freq in cp] for cp in corpus[:1]]

# build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus,
                                                id2word = id2word,
                                                num_topics = 50,
                                                random_state = 100,
                                                update_every = 1,
                                                chunksize = 100,
                                                passes = 5,
                                                alpha = 'auto',
                                                per_word_topics = True)
    # lda_model.print_topics()
print(lda_model.print_topics())

'''
[(39,
  '0.117*"cohen" + 0.113*"qualify" + 0.018*"snub" + 0.000*"lobos" + '
  '0.000*"jazzy" + 0.000*"prediction_market" + 0.000*"pivit" + '
  '0.000*"powerbroker" + 0.000*"mccue" + 0.000*"defensive_crouch"'),
 (33,
  '0.080*"israel" + 0.060*"embassy" + 0.056*"netanyahu" + 0.050*"israeli" + '
  '0.048*"jerusalem" + 0.046*"settlement" + 0.042*"palestinian" + '
  '0.039*"peace" + 0.033*"ambassador" + 0.024*"resolution"'),
 (19,
  '0.418*"student" + 0.052*"news" + 0.034*"underwater" + 0.033*"click" + '
  '0.032*"correspondent" + 0.032*"anchor" + 0.027*"transcript" + '
  '0.026*"classroom" + 0.023*"cnn" + 0.010*"today"'),
 (28,
  '0.310*"church" + 0.078*"christian" + 0.071*"god" + 0.059*"pastor" + '
  '0.032*"religion" + 0.029*"rev" + 0.026*"bible_study" + 0.022*"bishop" + '
  '0.019*"congregation" + 0.019*"spiritual"'),
 (12,
  '0.161*"sexual" + 0.147*"nasa" + 0.065*"migrant" + 0.062*"libya" + '
  '0.043*"expectation" + 0.042*"educate" + 0.039*"elderly" + 0.036*"shower" + '
  '0.028*"tunisian" + 0.025*"bangladesh"'),
 (27,
  '0.095*"japan" + 0.090*"cool" + 0.090*"india" + 0.074*"compete" + '
  '0.064*"honest" + 0.063*"japanese" + 0.049*"knight" + 0.043*"ford" + '
  '0.024*"jeffrey" + 0.016*"centre"'),
 (35,
  '0.053*"syria" + 0.046*"turkey" + 0.037*"syrian" + 0.031*"turkish" + '
  '0.030*"coalition" + 0.024*"fight" + 0.024*"kuwait" + 0.023*"rebel" + '
  '0.021*"airstrike" + 0.020*"conflict"'),
 (40,
  '0.100*"china" + 0.054*"chinese" + 0.027*"north_korea" + 0.021*"south_korea" '
  '+ 0.021*"launch" + 0.020*"country" + 0.019*"philippine" + 0.019*"satellite" '
  '+ 0.017*"nuclear_weapon" + 0.016*"international"'),
 (3,
  '0.104*"hotel" + 0.077*"tourist" + 0.061*"restaurant" + 0.054*"aviation" + '
  '0.054*"customer" + 0.053*"employee" + 0.043*"spokesperson" + 0.036*"dalla" '
  '+ 0.032*"vacation" + 0.030*"et"'),
 (42,
  '0.178*"clinton" + 0.078*"campaign" + 0.046*"sander" + 0.030*"email" + '
  '0.026*"aide" + 0.025*"hillary_clinton" + 0.024*"former" + 0.019*"committee" '
  '+ 0.019*"secretary" + 0.018*"democratic"'),
 (23,
  '0.044*"company" + 0.033*"business" + 0.028*"work" + 0.025*"worker" + '
  '0.020*"sell" + 0.018*"trade" + 0.015*"industry" + 0.014*"market" + '
  '0.014*"value" + 0.013*"interest"'),
 (2,
  '0.033*"government" + 0.022*"country" + 0.022*"escape" + 0.018*"year" + '
  '0.011*"release" + 0.010*"kill" + 0.010*"group" + 0.010*"inmate" + '
  '0.009*"april" + 0.009*"authority"'),
 (46,
  '0.031*"world" + 0.026*"story" + 0.024*"write" + 0.023*"book" + 0.015*"life" '
  '+ 0.015*"read" + 0.014*"photograph" + 0.013*"brown" + 0.013*"become" + '
  '0.011*"facebook"'),
 (36,
  '0.019*"run" + 0.019*"first" + 0.016*"take" + 0.015*"see" + 0.013*"look" + '
  '0.012*"man" + 0.012*"hand" + 0.012*"side" + 0.011*"minute" + 0.011*"head"'),
 (7,
  '0.063*"say" + 0.035*"city" + 0.024*"people" + 0.022*"state" + 0.021*"area" '
  '+ 0.020*"cnn" + 0.018*"accord" + 0.012*"saturday" + 0.011*"local" + '
  '0.011*"official"'),
 (13,
  '0.029*"american" + 0.023*"people" + 0.019*"country" + 0.015*"america" + '
  '0.014*"many" + 0.011*"make" + 0.010*"community" + 0.009*"even" + '
  '0.009*"political" + 0.009*"nation"'),
 (15,
  '0.049*"family" + 0.037*"child" + 0.022*"life" + 0.018*"die" + '
  '0.017*"mother" + 0.017*"father" + 0.016*"home" + 0.015*"school" + '
  '0.015*"son" + 0.014*"young"'),
 (5,
  '0.026*"year" + 0.022*"may" + 0.019*"new" + 0.019*"use" + 0.014*"find" + '
  '0.011*"could" + 0.010*"high" + 0.009*"even" + 0.009*"number" + '
  '0.009*"make"'),
 (47,
  '0.050*"say" + 0.035*"go" + 0.032*"get" + 0.023*"know" + 0.020*"people" + '
  '0.019*"think" + 0.017*"want" + 0.015*"see" + 0.015*"make" + 0.015*"time"'),
 (49,
  '0.075*"say" + 0.018*"cnn" + 0.013*"would" + 0.012*"tell" + 0.011*"call" + '
  '0.009*"take" + 0.009*"week" + 0.009*"time" + 0.008*"statement" + '
  '0.008*"last"')]
'''

doc_lda = lda_model[corpus]
pprint(doc_lda[0])
    
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

lda = models.LdaModel(corpus_tfidf, id2word = id2word, num_topics = 20) 
corpus_lda = lda[corpus_tfidf]
lda_csc_matrix = matutils.corpus2csc(corpus_lda).transpose()  #gensim sparse matrix to scipy sparse matrix
lda_csc_matrix.shape
    
# top terms in docs
# use top terms in topics of each docs to form new vocabulary
# do tf-idf using new vocabulary
topics_terms = lda_model.state.get_lambda() 
#convert estimates to probability (sum equals to 1 per topic)
topics_terms_proba = np.apply_along_axis(lambda x: x/x.sum(),1,topics_terms)
# find the right word based on column index
words = [lda_model.id2word[i] for i in range(topics_terms_proba.shape[1])]
#put everything together
term = pd.DataFrame(topics_terms_proba,columns=words)

'''
##########
'''
#------------- test -------------

# Total cnn News
df = cnn1

data_lemmatized = nlp_process(df)

doc_lda_total, corpus_tfidf_total, topics_terms_total, topics_terms_proba_total, term_total = get_topic_word(data_lemmatized)

# divide cnn news by year
cnn_year = []
for y in range(2012,2018):
    cnn_year.append(cnn1[cnn1["year"] == y]) 


