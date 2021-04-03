import time
import re
import string
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import copy


browser = webdriver.Chrome(r"C:\Users\Mert\Downloads\chromedriver_win32\chromedriver.exe")


browser.get("https://www.trendyol.com/sima/50-adet-3-katli-yumusak-burun-telli-full-ultrasonik-renkli-cerrahi-maske-p-46143528/yorumlar?boutiqueId=559205&merchantId=188497")

time.sleep(5)

elem = browser.find_element_by_tag_name("body")
comments = pd.DataFrame(columns = ['comments'])
no_of_pagedowns = 520

while no_of_pagedowns:
    elem.send_keys(Keys.PAGE_DOWN)
    time.sleep(1)
    print(no_of_pagedowns)
    no_of_pagedowns-=1

post_elems = browser.find_elements_by_class_name("rnr-com-tx")
print(len(post_elems))

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F1F2-\U0001F1F4"  # Macau flag
        u"\U0001F1E6-\U0001F1FF"  # flags
        u"\U0001F600-\U0001F64F"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U0001F1F2"
        u"\U0001F1F4"
        u"\U0001F620"
        u"\u200d"
        u"\u2640-\u2642"
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

for post in post_elems:
     comment = post.text
     comment = comment.lower()
                
     comment = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", comment)
     #comment = unidecode.unidecode(comment)   
     comment = comment.translate(str.maketrans('', '', string.punctuation))
     comment = ''.join([i for i in comment if not i.isdigit()])
     comment = deEmojify(comment)
     comment = comment.strip()
     comments.loc[len(comments)] = [comment]
     
     
     
     
     
comments_copy = copy.deepcopy(comments)   
     
def remove_space(s):
    return s.replace("\n"," ")



comments_copy['comments'] = comments_copy['comments'].apply(remove_space)

comments_copy.to_csv('goodDataset2.csv',encoding='utf-8-sig', header=True, sep=',',mode='a',index=False) #index=False yapılarak 1 2 3 önlenebilir

browser.quit()        
     