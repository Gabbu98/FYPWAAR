import numpy as np
import pandas as pd

import sklearn
from sklearn.decomposition import TruncatedSVD

columns = ['id','taskID','Rating']
frame = pd.read_csv('DataVisitor.csv',sep='\t', names=columns)
print(frame.head())