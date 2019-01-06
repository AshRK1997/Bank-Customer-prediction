import pandas as pd
import numpy as np
bank = pd.read_csv('/home/ashwinrk/Desktop/simle_bank.csv')

#remove features useless in the data

bank.drop(['RowNumber','CustomerId','Surname'],axis = 1, inplace= True) #axis = 1 means it searches coloumnwise (default is searching rowwise) ,  inplace ="true" means deleting it permanently

bank2 = bank.copy()

# encode text to number since dataset has strings in it which cannot be processed its called encoding

from sklearn.preprocessing import LabelEncoder

#encoding geography

le1 = LabelEncoder()
le1.fit(bank.Geography)
bank.Geography = le1.transform(bank.Geography)

#encoding Gender
le2 = LabelEncoder()
le2.fit(bank.Gender)
bank.Gender = le2.transform(bank.Gender)

#onehot encoder is used to convert geographical part which is continous to discrete/binary/categorical

from sklearn.preprocessing import OneHotEncoder

x = bank.drop(['Exited'],axis=1) #x is input but exited is the answer so we remove it
y = bank['Exited']             # y is the output which is exited

x = np.array(x)
y = np.array(y)

ohe = OneHotEncoder(categorical_features=[1]) #one hot encoder takes only in numpy frame and categorical feature is the coloumn 1(1st coloum is coloumn0)

ohe.fit(x)

x = ohe.transform(x).toarray()

from sklearn.model_selection import train_test_split

#in this part we are using 80% of data is used to train and 20% is used to test

xtr, xts, ytr, yts = train_test_split(x,y,test_size=0.2)

#scale the data

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(xtr)
xtr = sc.transform(xtr)
xts = sc.transform(xts)

#applying neural network algorithm

from sklearn.neural_network import MLPClassifier
# hidden ayers are decided by us and inut output layer decided by dataset . if we want 2 hidden layers of 5 neurons then(5,5) if 3 hidden layers of 5 neurons then (5,5,5) also verbose means it shows the cost function on console
#max_iter means maximum number of iteration is
alg = MLPClassifier(hidden_layer_sizes=(5,5),max_iter=1000)

#training the networking
alg.fit(xtr,ytr)

#checking the accuracy

accuracy = alg.score(xts,yts)
print(accuracy)


ip = np.array([1,0,0,590,1,38,3,84250,2,1,1,100000.0]).reshape(1,12)
out = alg.predict(ip)
print(out)

#recalling

yp = alg.predict(xts)
from sklearn import metrics

recallrate = metrics.recall_score(yts,yp)
print(recallrate)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,4))
sns.distplot(bank.CreditScore[bank.Exited == 1])
sns.distplot(bank.CreditScore[bank.Exited == 0]) #change bank.Creditscore to bank.age to get graph
plt.show()

