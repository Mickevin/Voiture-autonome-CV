import pickle
from flask import Flask, render_template, request
from sklearn.svm import SVR
from sklearn.datasets import fetch_california_housing
from  sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

model = SVR(kernel = "rbf")

df = fetch_california_housing()
X = pd.DataFrame(df['data'])
y = np.array(df['target'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
model.fit(X_train, y_train)

with open("model.sav", 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

app = Flask(__name__)

@app.route('/')
def california_index(): 
    return render_template('index.html')

@app.route('/predict/', methods=['POST'])
def result():
   
    if request.method == 'POST':
        a = float(request.form['MedInc'])
        b = float(request.form['HouseAge'])
        c = float(request.form['AveRooms'])
        d = float(request.form['AveBedrms'])
        e = float(request.form['Population'])
        f = float(request.form['AveOccup'])
        g = float(request.form['Latitude'])
        h = float(request.form['Longitude'])

        data = np.array([[a, b, c, d, e, f, g, h]])

        model = pickle.load(open(f"model.sav", 'rb'))[0]

        pred = model.predict(data)
        return render_template('prediction.html', price=pred)#int(pred))



if __name__ == '__main__':
    app.debug = True
    app.run(
        host='localhost',
        port = 5000,
        debug = True)