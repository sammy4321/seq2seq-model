# -*- coding: utf-8 -*-
import nltk
example_text='''Environment consists of both living and non-living things that surround us. There is a relationship between living beings and the environment.

There is no doubt for the fact that man is an intelligent animal. Man is able to influence environment with his activities. The effects of man's activities on environment have been both positive and negative.

On one hand, the scientific inventions of man has made human life comfortable. On the other hand, these inventions have caused danger for the sustainability of our environment.

We travel in vehicles that uses petroleum and emits harmful gases causing air pollution. Many of the things of our everyday life such as lights, refrigerator, fans, television, etc. run on electricity. However, the fossil fuels that are burned to produce electricity causes an increase in greenhouse gases. These gases trap the temperature and don't allow it to get released. As a result, the planet earth's temperature increases that damage the environment. This increased in earth's temperature is called global warming.

Human-beings are most powerful and intelligent among all living creatures. Thus, it is the responsibility of every human-being to protect and save the environment, so that the future generation may enjoy the gifts of nature and environment.'''
from nltk.tokenize import sent_tokenize,word_tokenize
all_words=word_tokenize(example_text)
#all_words=example_text.split()
print(all_words)
uniqueWords=[]
for i in all_words:
	if i not in uniqueWords :
		uniqueWords.append(i)

import numpy as np
uniqueWords_integer=np.arange(len(uniqueWords))

all_words_in_integer=[]
for i in all_words:
	all_words_in_integer.append(uniqueWords_integer[uniqueWords.index(i)])

print(all_words_in_integer)

#Padding to same size start

a=[[1,2,3],[1,2],[1,2,3,4,5],[1]]
longest=len(a[0])
for i in a:
	if len(i) > longest:
		longest=len(i)
print(longest)
for i in a:
	for j in range(longest-len(i)):
		i.append(0)

for i in a:
	print i