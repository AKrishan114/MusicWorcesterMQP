import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import math
from pomegranate import *
import pybbn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import itertools

new_d1 = pd.read_csv('211Clustering - LatestMetrics.csv')

new_d1 = new_d1[new_d1.Segment != 'Best']
new_d1 = new_d1[new_d1.Segment != 'One&Done']
new_d1 = new_d1[new_d1.Segment != 'New']
new_d1 = new_d1[new_d1.Segment != 'High']
new_d1 = new_d1[new_d1.Subscriber != 'Current']

new_d1["Subscriber"] = new_d1["Subscriber"].map({"previous": 1, "never": 0})

subscriber_counts = new_d1.groupby("Cluster KMeans")["Subscriber"].sum()
print(subscriber_counts)