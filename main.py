from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
warnings.filterwarnings('ignore')
from feature import generate_data_set
# Gradient Boosting Classifier Model

import pickle
from xgboost import XGBClassifier


data = pd.read_csv("urldata2.csv")


# Splitting the dataset into dependant and independant fetature
data = data.drop(['Protocol', 'Domain', 'Path'], axis = 1).copy()
y = data["label"]
X = data.drop(["label"],axis =1)


# instantiate the model

# instantiate the model
xbc = XGBClassifier()
xbc.fit(X,y)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", xx= -1)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        url = request.form["url"]
        x = np.array(generate_data_set(url)).reshape(1,13) 
        y_pred =xbc.predict(x)[0]
        y_pro_non_phishing = xbc.predict_proba(x)[0,1]
        #0 is safe       
        #1 is unsafe
        
        return render_template('index.html',xx =y_pred,yy =y_pro_non_phishing,url=url )
        
       


if __name__ == "__main__":
    app.run(debug=True)