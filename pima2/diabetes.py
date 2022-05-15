import os

import googleapiclient.discovery
import numpy as np
import pandas as pd
from flask_bootstrap import Bootstrap
from sklearn.preprocessing import MinMaxScaler

# to make this notebook's output stable across runs

np.random.seed(42)

from flask import Flask, render_template

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap = Bootstrap(app)

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired


class LabForm(FlaskForm):
    preg = StringField("# Pregnancies", validators=[DataRequired()])
    glucose = StringField("Glucose", validators=[DataRequired()])
    blood = StringField("Blood pressure", validators=[DataRequired()])
    skin = StringField("Skin thickness", validators=[DataRequired()])
    insulin = StringField("Insulin", validators=[DataRequired()])
    bmi = StringField("BMI", validators=[DataRequired()])
    dpf = StringField("DPF Score", validators=[DataRequired()])
    age = StringField("Age", validators=[DataRequired()])
    submit = SubmitField('Submit')


@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        X_test = np.array([[float(form.preg.data),
                            float(form.glucose.data),
                            float(form.blood.data),
                            float(form.skin.data),
                            float(form.insulin.data),
                            float(form.bmi.data),
                            float(form.dpf.data),
                            float(form.age.data)]])
        print(X_test.shape)
        print(X_test)

        data = pd.read_csv('./diabetes.csv', sep=',')

        X = data.values[:, 0:8]
        y = data.values[:, 8]

        scaler = MinMaxScaler()
        scaler.fit(X)

        X_test = scaler.transform(X_test)

        MODEL_NAME = "my_pima_model"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "neural-cortex-350305-9e53191bfbc5.json"
        project_id = "myproject-1548545104898"
        model_id = "my_mnist_model"
        model_path = "projects/{}/models/{}".format(project_id, model_id)
        model_path += "/versions/v0001/"  # if you want to run a specific version
        ml_resource = googleapiclient.discovery.build("ml", "v1").projects()

        input_data_json = {"signature_name": "serving_default", "instances": X.tolist()}
        request = ml_resource.predict(name=model_path, body=input_data_json)
        response = request.execute()
        if "error" in response:
            raise RuntimeError(response["error"])

        predD = np.arra([pred['dense_2'] for pred in response["predictions"]])
        print(predD[0][0])
        res = predD[0][0]

        return render_template('result.html', res=res)

    return render_template('prediction.html', form=form)


if __name__ == '__main__':
    app.run()
