from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import numpy as np
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.preprocessing import StandardScaler
#from sklearn.externals import joblib
#import pickle

# load the model from disk

rf = pickle.load(open('rf_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def diab():
	return render_template('diab.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        age = int(request.form['age'])
       
        data1 = np.array([[preg, glucose, bp, st, insulin, bmi, age]])

        my_prediction1 = rf.predict(data1)
        
    return render_template('diab_result.html', prediction=my_prediction1)

if __name__ == '__main__':
	app.run()

