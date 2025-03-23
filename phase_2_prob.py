import pickle
import pandas as pd
import time
F1=open("pickle_objs/Tech_Skills_List.pkl", 'rb')
techsk_repetition_list = pickle.load(F1)
F2=open("pickle_objs/Tech_Skills_Vocab.pkl", 'rb')
techsk_vocab = pickle.load(F2)
l=len(techsk_repetition_list)
l2=len(techsk_vocab)
Prob_Val={}
for skill in techsk_vocab:
    ct=techsk_repetition_list.count(skill)
    prob=ct/l
    Prob_Val[skill]=prob
Prob_Val_Sorted = {k: v for k, v in sorted(Prob_Val.items(), key= lambda v: v[1], reverse=True)}
Prob_File=open("pickle_objs/Probability.pkl", 'wb')
pickle.dump(Prob_Val_Sorted, Prob_File)
# select the three letters and give the corresponding words.if the required amount of words is no there use lavenstein.
#also use probability to determine the order of tfidf values. use levenstein by taking the word with highest
    #tfidf value. use the lavenstein function with the incomplete words of the tfidf.
    #then select the technical skills accordingly and arrange the words in the order of their probability.

