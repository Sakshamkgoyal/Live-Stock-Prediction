from numpy.random import RandomState
import pandas as pd

df = pd.read_csv('livestock.csv')
rng = RandomState()

train = df.sample(frac=0.7, random_state=rng)
test = df.loc[~df.index.isin(train.index)]

train.to_csv('/home/saksham/Stock/train.csv')
test.to_csv('/home/saksham/Stock/test.csv')