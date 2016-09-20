from __future__ import print_function

import sys
import os
import numpy as np

from pyspark import SparkContext 
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils 
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import DenseVector
from pyspark.mllib.feature import StandardScaler


def RF(trainDataPath, valDataPath, stardIdx, endIdx, isScaling, numTrees, maxDepth):
    trainData = sc.textFile(trainDataPath).map(lambda line: line.split(','))
    valData = sc.textFile(valDataPath).map(lambda line: line.split(',')) 
    trainLabel = trainData.map(lambda row: int(row[-1])-1)
    # RGBD: [0:4], EMP: [4:14], EAP:[14:41], FCN:[41:47]
    trainFeatures = trainData.map(lambda row: row[startIdx:endIdx]) 
    valLabel = valData.map(lambda row: int(row[-1])-1)
    valFeatures = valData.map(lambda row: row[startIdx:endIdx])
    if isScaling:
        scaler = StandardScaler(withMean=True, withStd=True).fit(trainFeatures)
        trainFeatures1 = scaler.transform(trainFeatures)
        trainData1 = trainLabel.zip(trainFeatures1).map(lambda (l,f): LabeledPoint(l, f))
    else:
        trainData1 = trainLabel.zip(trainFeatures).map(lambda (l,f): LabeledPoint(l,f))

    model = RandomForest.trainClassifier(trainData1, numClasses=6, categoricalFeaturesInfo={}, numTrees=numTrees, featureSubsetStrategy="auto", impurity='gini', maxDepth=maxDepth, maxBins=32)
    
    if isScaling:
        valFeatures1 = scaler.transform(valFeatures)
        valData1 = valLabel.zip(valFeatures1).map(lambda (l,f): LabeledPoint(l,f))
    else: 
        valData1 = valLabel.zip(valFeatures).map(lambda (l,f): LabeledPoint(l,f))

    predictions = model.predict(valData1.map(lambda x: x.features))
    labelsAndPredictions = valData1.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(valData1.count())
    print('Test Error = ' + str(testErr))


if __name__ == "__main__":
    trainDataPath = sys.argv[1]
    valDataPath = sys.argv[2]
    startIdx = int(sys.argv[3])
    endIdx = int(sys.argv[4])
    isScaling = int(sys.argv[5])
    numTrees = int(sys.argv[6])
    maxDepth = int(sys.argv[7])
    sc = SparkContext(appName="Classification using Spark Random Forest")
    RF(trainDataPath, valDataPath, startIdx, endIdx, isScaling, numTrees, maxDepth)

