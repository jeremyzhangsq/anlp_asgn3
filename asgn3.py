from __future__ import division
import math as mt
import operator
from load_map import wid2word, word2wid
from nltk.stem import *
from nltk.stem.porter import *
import matplotlib.pyplot as plt

STEMMER = PorterStemmer()

# helper function to get the count of a word (string)
def w_count(word):
  return o_counts[word2wid[word]]

def tw_stemmer(word):
  '''Stems the word using Porter stemmer, unless it is a 
  username (starts with @).  If so, returns the word unchanged.

  :type word: str
  :param word: the word to be stemmed
  :rtype: str
  :return: the stemmed word

  '''
  if word[0] == '@' or word[0] == '#': # don't stem these
    return word
  else:
    return STEMMER.stem(word)

def PMI(c_xy, c_x, c_y, N):
  '''Compute the pointwise mutual information using cooccurrence counts.

  :type c_xy: int 
  :type c_x: int 
  :type c_y: int 
  :type N: int
  :param c_xy: coocurrence count of x and y
  :param c_x: occurrence count of x
  :param c_y: occurrence count of y
  :param N: total observation count
  :rtype: float
  :return: the pmi value

  '''
  return mt.log2(N*c_xy/float(c_x*c_y)) # you need to fix this

# Do a simple error check using value computed by hand
if(PMI(2,4,3,12) != 1):
    # these numbers are from our y,z example
    print("Warning: PMI is incorrectly defined")
else:
    print("PMI check passed")

def cos_sim(v0,v1):
  '''
  Compute the cosine similarity between two sparse vectors.

  :type v0: dict
  :type v1: dict
  :param v0: first sparse vector
  :param v1: second sparse vector
  :rtype: float
  :return: cosine between v0 and v1
  '''
  # We recommend that you store the sparse vectors as dictionaries
  # with keys giving the indices of the non-zero entries, and values
  # giving the values at those dimensions.

  #You will need to replace with the real function
  sum0 = get_vector_len(v0)
  sum1 = get_vector_len(v1)
  total = 0
  for w in v0:
      if w in v1:
          total += v0[w]*v1[w]          
  return total/float(sum0*sum1)

def get_vector_len(v):
    return mt.sqrt(sum([i**2 for i in v.values()]))

def jaccard_sim(v0,v1):
    all_id = set()
    for i in v0.keys():
        all_id.add(i)
    for i in v0.keys():
        all_id.add(i)
    x = 0
    y = 0
    for id in all_id:
        if id not in v0.keys():
            x += 0
            y += v1[id]
        elif id not in v1.keys():
            x += 0
            y += v0[id]
        else:
            x+=min(v1[id],v0[id])
            y+=max(v1[id],v0[id])
    return x/y
    



def create_ppmi_vectors(wids, o_counts, co_counts, tot_count):
    '''Creates context vectors for the words in wids, using PPMI.
    These should be sparse vectors.

    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''
    vectors = {}
    for wid0 in wids:
        ## you will need to change this
        vectors[wid0] = {}
        cx = o_counts[wid0]
        for y in co_counts[wid0]:
            cxy = co_counts[wid0][y]
            cy = o_counts[y]
            vectors[wid0][y] = PMI(cxy,cx,cy,N)
            
    return vectors

def read_counts(filename, wids):
  '''Reads the counts from file. It returns counts for all words, but to
  save memory it only returns cooccurrence counts for the words
  whose ids are listed in wids.

  :type filename: string
  :type wids: list
  :param filename: where to read info from
  :param wids: a list of word ids
  :returns: occurence counts, cooccurence counts, and tot number of observations
  '''
  o_counts = {} # Occurence counts
  co_counts = {} # Cooccurence counts
  fp = open(filename)
  N = float(next(fp))
  for line in fp:
    line = line.strip().split("\t")
    wid0 = int(line[0])
    o_counts[wid0] = int(line[1])
    if(wid0 in wids):
        co_counts[wid0] = dict([int(y) for y in x.split(" ")] for x in line[2:])
  return (o_counts, co_counts, N)

def print_sorted_pairs(similarities, o_counts, first=0, last=100):
  '''Sorts the pairs of words by their similarity scores and prints
  out the sorted list from index first to last, along with the
  counts of each word in each pair.

  :type similarities: dict 
  :type o_counts: dict
  :type first: int
  :type last: int
  :param similarities: the word id pairs (keys) with similarity scores (values)
  :param o_counts: the counts of each word id
  :param first: index to start printing from
  :param last: index to stop printing
  :return: none
  '''
  if first < 0: last = len(similarities)
  for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse = True)[first:last]:
    word_pair = (wid2word[pair[0]], wid2word[pair[1]])
    print("{:.2f}\t{:30}\t{}\t{}".format(similarities[pair],str(word_pair),
                                         o_counts[pair[0]],o_counts[pair[1]]))

def freq_v_sim(sims):
  xs = []
  ys = []
  for pair in sims.items():
    ys.append(pair[1])
    c0 = o_counts[pair[0][0]]
    c1 = o_counts[pair[0][1]]
    xs.append(min(c0,c1))
  plt.clf() # clear previous plots (if any)
  plt.xscale('log') # set x axis to log scale. Must do *before* creating plot
  plt.plot(xs, ys, 'k.') # create the scatter plot
  plt.xlabel('Min Freq')
  plt.ylabel('Similarity')
  print("Freq vs Similarity Spearman correlation = {:.2f}".format(spearmanr(xs,ys)[0]))
#  plt.show() #display the set of plots

def make_pairs(items):
  '''
  Takes a list of items and creates a list of the unique pairs
  with each pair sorted, so that if (a, b) is a pair, (b, a) is not
  also included. Self-pairs (a, a) are also not included.

  :type items: list
  :param items: the list to pair up
  :return: list of pairs

  '''
  return [(x, y) for x in items for y in items if x < y]


test_words = ["cat", "dog", "mouse", "computer","@justinbieber"]
stemmed_words = [tw_stemmer(w) for w in test_words]
all_wids = set([word2wid[x] for x in stemmed_words]) # stemming might create duplicates; remove them

# you could choose to just select some pairs and add them by hand instead
# but here we automatically create all pairs 
wid_pairs = make_pairs(all_wids)


# read in the count information
(o_counts, co_counts, N) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/counts", all_wids)

# make the word vectors
vectors = create_ppmi_vectors(all_wids, o_counts, co_counts, N)

# compute cosine similarites for all pairs we consider
c_sims = {(wid0,wid1): cos_sim(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}

print("Sort by cosine similarity")
print_sorted_pairs(c_sims, o_counts)

# compute cosine similarites for all pairs we consider
j_sims = {(wid0,wid1): jaccard_sim(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}

print("Sort by jaccard similarity")
print_sorted_pairs(j_sims, o_counts)