from flask import Flask,request,render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model_regression = pickle.load(open('models/MODEL_REGRESSION.pkl','rb'))
model_scaled = pickle.load(open('models/MODEL_SCALED.pkl','rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        MedInc = float(request.form.get('MedInc'))
        HouseAge = float(request.form.get('HouseAge'))
        AveRooms = float(request.form.get('AveRooms'))
        AveBedrms = float(request.form.get('AveBedrms'))
        Population = float(request.form.get('Population'))
        AveOccup = float(request.form.get('AveOccup'))
        Latitude = float(request.form.get('Latitude'))
        Longitude = float(request.form.get('Longitude'))
        
        model_ready = model_scaled.transform([[MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude]])
        final_result = model_regression.predict(model_ready) 
        # pass
        return render_template('home.html',result = final_result[0])
    
    else:
        return render_template("home.html")
    
    
if __name__=="__main__":
    app.run(host="0.0.0.0")
