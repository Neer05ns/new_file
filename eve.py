from flask import Flask, request,url_for, render_template #flask class #flask module
import pandas as pd
import numpy as np
import pickle

logistic_model=pickle.load(open("new.pkl","rb"))

app =  Flask(__name__)


#column_list= pickle.load(open("column.pkl", "rb"))

@app.route('/')
def home_page():
    return render_template("home.html")
    

@app.route('/prediction',methods=["POST"])
def prediction():
    
    Age = int(request.form["Age"])
    EstimatedSalary = int(request.form["EstimatedSalary"])

    arr=np.array([[Age,EstimatedSalary]])
    

    prediction = logistic_model.predict(arr)


    return render_template("welcome.html",data=prediction)
if __name__ == "__main__":
    app.run()
