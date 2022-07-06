import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Training the model, import classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# import evaluation metrics
from sklearn.model_selection import train_test_split


df = pd.read_csv("customer_data.csv")
df.fillna(value= np.mean(df), inplace= True)

X = df.drop(labels= 'label', axis= 1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.17, random_state= 100)

algorithm_instance = {
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'GaussianNB': GaussianNB(),
    'LogisticRegression': LogisticRegression(),
    'RandomForestClassifier': RandomForestClassifier(),
    'SVC': SVC()
}

models_score = {}
for alg_name, algo in algorithm_instance.items():
    fit_model = algo.fit(X_train, y_train)
    models_score[alg_name] = fit_model.score(X_test, y_test)

model_df = pd.DataFrame(data= models_score.values(), columns= ['scores'], index= models_score.keys())

fig, ax = plt.subplots(figsize= (15, 5))
ax.bar(x= model_df.index, height= model_df['scores'], label= 'scores')
plt.xticks(rotation= 360)
plt.legend()
plt.savefig('model_performance.jpg')
