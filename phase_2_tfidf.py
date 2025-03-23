import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
F3 = open("pickle_objs/ROWS.pkl","rb")
all_rows = pickle.load(F3)
#print(all_rows)
vectorizer=TfidfVectorizer(lowercase=False)
vectors=vectorizer.fit_transform(all_rows)
feature_names=vectorizer.get_feature_names_out()
dense=vectors.todense()
denselist=dense.tolist()
df=pd.DataFrame(denselist, columns=feature_names) #takes the tfidf value for each row and appends it
Row_Specific_Tfidfs = df.to_dict("list")
Avg_Tfidf={}
for word,tfidfs in Row_Specific_Tfidfs.items():
    count_of_0=tfidfs.count(0.0)
    ele=len(tfidfs)-count_of_0
    mean=sum(tfidfs)/ele
    Avg_Tfidf[word]=mean
#print(Avg_Tfidf)
Avg_Tfidf_Sorted = {k: v for k, v in sorted(Avg_Tfidf.items(), key= lambda v: v[1], reverse=True)}
Tfidf_File=open("pickle_objs/Avg_Tfidf.pkl", 'wb')
pickle.dump(Avg_Tfidf_Sorted, Tfidf_File)