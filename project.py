import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

data = {
    'Hours':[2,4,6,8,10],
    'Marks':[20,40,60,80,100]
}

df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Marks']

model = LinearRegression()
model.fit(X,y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))