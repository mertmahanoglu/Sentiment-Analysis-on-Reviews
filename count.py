# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:59:26 2021

@author: Mert
"""
import csv
import codecs
import shutil
"""
with codecs.open("C:/Users/Mert/Desktop/Datasets/birleştirilmiş/goodDataset1.csv", encoding="latin-1") as input_file:
    with codecs.open(
            "C:/Users/Mert/Desktop/Datasets/birleştirilmiş/goodDataset12.csv", "w", encoding="latin-1") as output_file:
        shutil.copyfileobj(input_file, output_file)"""


#C:/Users/Mert/.spyder-py3/Udemy/badDataset2.csv
#C:/Users/Mert/.spyder-py3/Udemy/goodDataset2.csv
#TrimmedBad1------------- TrimmedGood2

with open('C:/Users/Mert/Desktop/Datasets/birleştirilmiş/bot.csv', "r+", encoding="utf-8-sig") as csv_file:
    content = csv_file.read()

with open('C:/Users/Mert/Desktop/Datasets/birleştirilmiş/bot.csv', "w+",encoding="utf-8-sig") as csv_file:
    csv_file.write(content.replace('”', ''))
        


with open('C:/Users/Mert/Desktop/Datasets/birleştirilmiş/bot.csv', 'r+',encoding="utf-8-sig") as inf, open('C:/Users/Mert/Desktop/Datasets/birleştirilmiş/bot2.csv', 'w+', encoding="utf-8-sig") as of:
    for line in inf:
        trim = (field.strip() for field in line.split(','))
        of.write(','.join(trim)+'\n')