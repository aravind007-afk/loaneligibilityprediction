from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def mainn():
    return render_template("loan.html")

@app.route('/predict',methods=['post'])
def predict():
    # int_features=[int(x) for x in request.form.values()]
    features=[]
    for x in request.form.values():
        print(x)
        if x=='graduate' or x=='no':
            features.append(0)
        elif x=='not-graduate' or x=='yes':
            features.append(1)
        else:
            features.append(int(x))
    final=[np.array(features)]
    print(features)
    print(final)
    prediction=model.predict(final)
    #output='{0:.{1}f}'.format(prediction[0][1], 2)

    if prediction==0:
        return render_template('loan.html',pred='Loan approved')
    else:
        return render_template('loan.html',pred='Loan is not approved')


if __name__ == '__main__':
    app.run()