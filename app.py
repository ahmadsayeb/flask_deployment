from flask import Flask, render_template, request,jsonify
import flask
import numpy as np
import pickle
from joblib import load

#Defining The app
app = Flask(__name__)

#import model
classifier = load('rf_model_job')


#defining the route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features_array = np.array(features)
    features_reshaped = features_array.reshape(1,-1)
    prediction = classifier.predict(features_reshaped)

    output = int(prediction)
    if output == 1:
        output_text = 'Approved for the loan'
    else:
        output_text = 'Did not aprrove for the loan'
    
    return render_template('index.html', prediction_text=output_text)

if __name__=='__main__':
    app.run(debug=True)