
# coding: utf-8

# In[152]:


import xml.sax
import numpy as np
import pandas as pd
import numpy
import copy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[153]:


class AuthorHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.CurrentData = ""
        self.personID = ""
        self.FullName = ""
        self.FirstName = ""
        self.LastName = ""
        self.title = ""
        self.year = ""
        self.authors = ""
        self.jconf = ""
        self.id = ""
        self.label = ""
        self.organization = ""
    #元素开始事件处理
    def startElement(self, tag, attributes):
        self.CurrentData = tag
       
    def endElement(self, tag):
        return

    def characters(self, content):

        if self.CurrentData == "personID":
            self.personID+=content
        elif self.CurrentData == "FullName":
            self.FullName+=content
        elif self.CurrentData == "FirstName":
            self.FirstName+=content
        elif self.CurrentData == "LastName":
            self.LastName+=content
        elif self.CurrentData == "title":
            self.title+=content
        elif self.CurrentData == "year":
            self.year+=content
        elif self.CurrentData == "authors":
            self.authors+=content
        elif self.CurrentData == "jconf":
            self.jconf+=content
        elif self.CurrentData == "label":
            self.label+=content
        elif self.CurrentData == "organization":
            self.organization+=content


# In[154]:


def Wipe_off_Punctuation(str_list):
    str_Wiped = []
    #去掉title中的标点符号,并且全部转化为小写
    Punctuation = [',', ':', '(', ')', '-', '_', ';', '.', '\'']
    for i in range(len(str_list)):
        temp = str_list[i]
        for j in range(len(Punctuation)):
            temp = temp.replace(Punctuation[j]," ").lower()
        str_Wiped.append(temp)
    return str_Wiped


# In[155]:


def Split_Title(data):
    data_split=copy.deepcopy(data)
    Vocab_table=""
    
    for i in range(len(data_split)):
        Vocab_table += (data_split[i])
        Vocab_table += " "
        data_split[i]=data_split[i].split()

    Vocab_table = Vocab_table.split()
    return Vocab_table,data_split


# In[156]:


def Split_Authors(data):
    #data_split,拆分后的data
    data_split = copy.deepcopy(data)
    #table是词表
    Vocab_table = ""
    
    for i in range(len(data_split)):
        Vocab_table += (data_split[i])
        Vocab_table += ","
        data_split[i]=data_split[i].split(",")

    Vocab_table = Vocab_table.split(',')
    return Vocab_table[:-1],data_split


# In[157]:


def One_hot_encoding(vocabulary,sequence): #生成对应one-hot
    vocabulary_one_hot = np.array(vocabulary)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(vocabulary_one_hot)
    sequence_encoded = []
    Max_Sequence_Len = 0
    vocab_size = max(integer_encoded) + 2 #留出一个0来做padding
    #构建one-hot词表    
    for i in range(len(sequence)):
        if Max_Sequence_Len < len(sequence[i]):
            Max_Sequence_Len = len(sequence[i])
        sequence_encoded.append(label_encoder.transform(sequence[i]) + 1)
    return sequence_encoded,Max_Sequence_Len,vocab_size


def Padding_One_hot(sequence_encoded,Max_Sequence_Len):
    One_hot_Padding = []
    for i in range(len(sequence_encoded)):
        Padding_ith_sequence = [0 for i in range(Max_Sequence_Len)] 
        Padding_ith_sequence[0:len(sequence_encoded[i])] = sequence_encoded[i]
        One_hot_Padding.append(Padding_ith_sequence)
    return One_hot_Padding

# In[158]:


def ProcessingRawData(FileName):
    # 创建一个 XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    # 重写 ContextHandler
    Handler = AuthorHandler()
    parser.setContentHandler( Handler )
    #parser.parse("AjayGupta1.xml")
    parser.parse(FileName)
    labels = Handler.label.split("\n\t\t")
    labels = list(map(int,labels[:-1]))
    jconf = Handler.jconf.split("\n\t\t")
    jconf = jconf[:-1]
    title = Handler.title.split("\n\t\t")
    title = title[:-1]
    authors = Handler.authors.split("\n\t\t")
    authors = authors[:-1]
    FullName = Handler.FullName.split("\n\t")
    FullName = FullName[0]
    organization = Handler.organization.split("\n\t\n\t")
    return title,labels,jconf,authors,FullName,organization
'''
# ## <b>读取原始数据：
title,labels,jconf,authors,FullName = ProcessingRawData("AjayGupta1.xml")
#title_list,vocabulary_one_hot,Max_sequence_len = One_hot_encoding(title)
# ## <b>对title进行清洗、拆分
title = Wipe_off_Punctuation(title)
title_vocab,title_split = Split_Title(title)
# ## <b>对title进行one-hot编码
title_one_hot,_ = One_hot_encoding(title_vocab,title_split)
# ## <b>对authors进行拆分
author_vocab,authors_split = Split_Authors(authors)
'''
