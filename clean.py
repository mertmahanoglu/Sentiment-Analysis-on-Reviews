# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 20:55:07 2021

@author: Mert
"""
"""
with open('C:/Users/Mert/Desktop/Datasets/birleştirilmiş/deneme.csv', "r+", encoding="utf-8-sig") as csv_file:
    content = csv_file.read()
    
with open('C:/Users/Mert/Desktop/Datasets/birleştirilmiş/deneme.csv', "w+",encoding="utf-8-sig") as csv_file:
    csv_file.write(content.replace('?', ''))"""
    
    
    #NoDuplicatesGood
    #NoDuplicatesBad
    #Duplicate removing
    #Shuffled
"""from more_itertools import unique_everseen
with open('C:/Users/Mert/Desktop/Datasets/birleştirilmiş/Shuffled.csv','r',encoding="utf-8-sig") as f, open('C:/Users/Mert/Desktop/Datasets/birleştirilmiş/3.csv','w',encoding="utf-8-sig") as out_file:
    out_file.writelines(unique_everseen(f))
"""


"""
with open('datasetML555.csv','r',encoding="utf-8-sig") as f, open('derlemtr2016-10000.txt','r',encoding="utf-8-sig") as f2:
   a=f.read().split();b=f2.read().split()
   new = ' '.join(i for i in a if i.lower() not in (x.lower() for x in b))   
   """
   
import re
import pandas as pd
def demoji(text):
	emoji_pattern = re.compile("["
		u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U00010000-\U0010ffff"
	                           "]+", flags=re.UNICODE)
	return(emoji_pattern.sub(r'', text))
 
data = pd.read_csv('goodDataset2.csv',encoding='utf-8-sig', sep='\t') # read tsv file
# data = pd.read_csv('test.csv',encoding='utf-8')  read csv file
 
data[u'comments'] = data[u'comments'].astype(str)
data[u'comments'] = data[u'comments'].apply(lambda x:demoji(x))
data.to_csv('output.csv',index=False, encoding='utf-8-sig') # save to csv file