import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__,static_url_path='/static')
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/')
@app.route('/')
@app.route('/first') 
def first():
	return render_template('first.html')
@app.route('/login') 
def login():
	return render_template('login.html')    
@app.route('/chart') 
def chart():
	return render_template('chart.html')    
@app.route('/abstract') 
def abstract():
	return render_template('abstract.html') 
@app.route('/future') 
def future():
	return render_template('future.html')  
@app.route('/upload') 
def upload():
	return render_template('upload.html') 
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)    

 
@app.route('/home')

def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    final_features = scaler.transform(final_features)    
    prediction = model.predict(final_features)
    y_probabilities_test = model.predict_proba(final_features)
    y_prob_success = y_probabilities_test[:, 1]
    print("final features",final_features)
    print("prediction:",prediction)
    output = round(prediction[0], 2)
    y_prob=round(y_prob_success[0], 3)
    print(output)

    if output == 0:
        return render_template('index.html', prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A BENIGN CANCER WITH PROBABILITY VALUE  {}'.format(y_prob))
    else:
         return render_template('index.html', prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A MALIGNANT CANCER WITH PROBABILITY VALUE  {}'.format(y_prob))
        
@app.route('/predict_api',methods=['POST'])
def predict_api():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)