#-*- coding=UTF-8
#基于贝叶斯决策理论的分类方法
#贝叶斯方法优于KNN需要大量的计算

from numpy import *

#过滤社区侮辱性文字

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 侮辱性文字, 0 正常文字
    return postingList,classVec

#建立文档词条
def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)  #集合合并
    return list(vocabSet)

#词集模型
def setOfWord2Vec(vocabList,inputSet):
    returnVec=zeros(len(vocabList))
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print "这个单词不在所有的单词向量里面"
    return  returnVec

#词袋模型
def bagOfword2VecMN(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return  returnVec

#朴素贝叶斯训练函数
def trainB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numwords=len(trainMatrix[0])
    #对于category为0,1 才可以使用sum
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=ones(numwords)
    p1Num=ones(numwords)

    p0Denom=2.0
    p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])

    p1Vect=log(p1Num/p1Denom)
    p0Vect=log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

#朴素贝叶斯分类函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():

    dataSet,classVec=loadDataSet()
    vocabList=createVocabList(dataSet)

    trainMat=[]
    for doc in dataSet:
        trainMat.append(setOfWord2Vec(vocabList,doc))

    p0V,p1V,pAb=trainB0(array(trainMat),array(classVec))
    testEntry=['love','my','dalmation']
    thisDoc=array(setOfWord2Vec(vocabList,testEntry))
    #计算贝叶斯分类结果
    result=classifyNB(thisDoc,p0Vec=p0V,p1Vec=p1V,pClass1=pAb)
    result="正常言论" if result==0 else "侮辱言论"
    print r"分类结果:",result

#testingNB()





