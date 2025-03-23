import pickle
#import pandas as pd
import time
import enchant # type: ignore
import json
n=1
F1=open("pickle_objs/Avg_Tfidf.pkl", 'rb')
F2=open("pickle_objs/Probability.pkl",'rb')
Avg_Tfidf=pickle.load(F1)
Prob_Val=pickle.load(F2)
#print(Prob_Val)
#print(Prob_Val.keys())
def find_levenschteinD(base,word,distances):
    distance=enchant.utils.levenshtein(base,word)
    distances[word]=distance
    return distances
def Sort_Distances_Add_Close_Words(d,lst,limit):
    Distance_Sorted = {k: v for k, v in sorted(d.items(), key=lambda v: v[1])}
    Close_Words = list(Distance_Sorted.keys())
    for a in range(limit - len(lst)):
        Close_Word=Close_Words[a]
        lst.append(Close_Word)
    return lst

while n>0:
    while Prob_Val:
        Type_letters = input("Enter three or more letters: ").lower()
        l=len(Type_letters)
        if l<3:
            print("There are less than 3 letters. Enter again")
            continue
        break

    Selected_words = []
    Distances={}
    base_word=""
    leftover_words=[]
    for word in Avg_Tfidf.keys():
        #if word in stopwords:
            #continue
        lower_word=word.lower()
        if Type_letters==lower_word[ :l]:
            Selected_words.append(word)
            base_word=Selected_words[0]
        elif not base_word:
            leftover_words.append(word)
        elif base_word:
            Distances = find_levenschteinD(base_word, word, Distances)
    if len(Selected_words)<4:
        for b in range(len(leftover_words)):
            word=leftover_words[b]
            if Selected_words==[]:
                Distances = find_levenschteinD(Type_letters, word, Distances)
            else:
                Distances = find_levenschteinD(base_word, word, Distances)
        Selected_words=Sort_Distances_Add_Close_Words(Distances,Selected_words,4)
    #print(Selected_words)
    #print(Close_Words)

    Tech_list = []
    for Selected_word in Selected_words:
        Skill_Distances = {}
        for skill in Prob_Val.keys():
            if skill not in Tech_list:
                Skill_Distances=find_levenschteinD(Selected_word,skill,Skill_Distances)
                if Selected_word in skill:
                    Tech_list.append(skill)
            if len(Tech_list)==10:
                break
    Tech_list=Sort_Distances_Add_Close_Words(Skill_Distances,Tech_list,10)
    print(json.dumps(Tech_list,indent=2))
    print()
    while Tech_list:
        Found_Skill=input("""Did you find your desired word?
        Enter yes/no here: """).lower()
        Repeat=True
        if Found_Skill=="yes":
            Repeat=False
        elif Found_Skill!="no":
            print("Enter yes/no only")
            continue
        break
    if Repeat==True:
        continue
    break




