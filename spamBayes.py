#-*- coding=UTF-8
#基于贝叶斯决策理论的分类方法
#贝叶斯方法优于KNN需要大量的计算
#垃圾邮件的检测

from numpy import *
import re
from os import listdir
from bayes import *
import random
#将输入的文本字符串分割成单词list
def textParse(bigString):
    listOfTokens=re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    docList=[]
    classList=[]
    fullText=[]

    filenameList1=listdir("email/spam")
    for name in filenameList1:
        wordList=textParse(open("email/spam/%s"%name).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

    filenameList2=listdir("email/ham")
    for name in filenameList2:
        wordList = textParse(open("email/ham/%s" % name).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList=createVocabList(docList)
    trainingSet=range(len(docList))

    testSet=[]
    for i in range(int(0.2*len(docList))):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V,p1V,pSpam=trainB0(array(trainMat),array(trainClasses))
    errorCount=0

    for  docIndex in testSet:
        wordVector=setOfWord2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print r"错误率:",float(errorCount/float(len(testSet)))
spamTest()

