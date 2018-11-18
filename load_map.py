'''
Author: Luke Shrimpton, Sharon Goldwater, Henry Thompson
Date: 2014-11-01, 2016-11-08
Copyright: This work is licensed under a Creative Commons
Attribution-NonCommercial 4.0 International License
(http://creativecommons.org/licenses/by-nc/4.0/): You may re-use,
redistribute, or modify this work for non-commercial purposes provided
you retain attribution to any previous author(s).
'''
fp = open("/afs/inf.ed.ac.uk/group/teaching/anlp/lab8/wid_word")
wid2word={}
word2wid={}
TOTAL_WORD = 0
for line in fp:
    widstr,word=line.rstrip().split("\t")
    wid=int(widstr)
    wid2word[wid]=word
    word2wid[word]=wid
    
def get_total_word_number(filename):
    
    fp = open(filename)
    a = float(next(fp))
    total = 0
    for line in fp:
        line = line.strip().split("\t")
        d = dict([int(y) for y in x.split(" ")] for x in line[2:])
        for i in d.keys():
            total += d[i]       
    return total/2
TOTAL_WORD = get_total_word_number("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/counts")