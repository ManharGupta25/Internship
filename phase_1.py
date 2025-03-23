#extracting the role and technical skills columns:-
import pandas as pd
import csv
import os
import pandas.core.series
import time
import pickle

file_df=pd.read_csv("C:/Users/manha/OneDrive/Desktop/taxonomy file/taxonomy_2500.csv", index_col="Role")
sub=file_df[["Technical_Skills"]]
os.mkdir("csv docs")
os.mkdir("pickle_objs")

#collecting all role names:-
sub.rename(index={"Transitions/Migration Manager": "Transitions_Migration Manager"}, inplace=True)
roles=set(sub.index)

#grouping technical skills based on roles:-
def generate_role_section(role):
    section = sub.loc[role]
    section_type = type(section)
    return section,section_type
def initialize_read_mode(role):
    file = open(f"csv docs/{role}.csv", 'r')
    file_read = csv.reader(file, delimiter=",")
    return file_read,file
def create_RowStr_append_to_allrows(row,all_rows):
    RowStr = ""
    l = len(row)
    for i in range(l):
        RowStr= RowStr+" "+row[i]
    all_rows.append(RowStr)
    return all_rows
def Pickle_Data(role,role_lst):
    role_pkl_obj = open(f"pickle_objs/{role}.pkl", 'wb')
    pickle.dump(role_lst, role_pkl_obj)
    role_pkl_obj.close()

Tech_Skills=[]
all_rows=[]
for role in roles:
    file=open(f"csv docs/{role}.csv", 'w') #making role-specific csv file and adding technical skill data
    Role_Section,section_type=generate_role_section(role)

    if section_type==pandas.core.series.Series:
        for Tech_Skills_Row in Role_Section:
            file.write(Tech_Skills_Row)
    else:
        Tech_Skills_Rows=Role_Section.to_string(index=False)
        file.write(Tech_Skills_Rows)

    file_read,file=initialize_read_mode(role)
    line_count=0
    role_lst=[]

    for row in file_read:
        if section_type==pandas.core.series.Series:
            pass
        elif line_count==0:
            line_count=1
            continue
        row[0]=row[0].lstrip()
        role_lst.extend(row) #combining different rows to make unified list containing data for resp. role
        all_rows=create_RowStr_append_to_allrows(row,all_rows)

    Pickle_Data(role,role_lst)
    Tech_Skills.extend(role_lst) #making common vocabulary contaning each single technical skill
    file.close()

vocab=set(Tech_Skills)
roles_pkl=open("pickle_objs/Roles.pkl", 'wb')
pickle.dump(roles, roles_pkl)
tech_skills_list=open("pickle_objs/Tech_Skills_List.pkl", 'wb')
pickle.dump(Tech_Skills, tech_skills_list)
tech_skills_vocab=open("pickle_objs/Tech_Skills_Vocab.pkl", 'wb')
pickle.dump(vocab, tech_skills_vocab)
all_rows_pkl=open("pickle_objs/ROWS.pkl", 'wb')
pickle.dump(all_rows, all_rows_pkl)
roles_pkl.close()
tech_skills_list.close()
tech_skills_vocab.close()
all_rows_pkl.close()
print(time.time())
#group by() for taking specific column
#pickle.format and depickle- for easy data transmission in a file
#time.time()- execution time
#sklearn()- tfidf calculation
#functions
# a set of technical skills
# list of technical skills with repetition
# iterating and finding the occurence of each skill.













