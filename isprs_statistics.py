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

if __name__ == "__main__":
    trainDataPath = sys.argv[1]
    valDataPath = sys.argv[2]
    sc = SparkContext(appName="Classification using Spark Random Forest")
    # trainData = sc.textFile(trainDataPath).map(lambda line: line.split(','))
    valData = sc.textFile(valDataPath).map(lambda line: line.split(',')) 
    # print(trainData.count())
    print(valData.count())

