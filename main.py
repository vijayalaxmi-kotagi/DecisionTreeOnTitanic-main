from flask import Flask, request, render_template
import joblib
import pandas as pd


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1000 * 1000


@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')


@app.route('/predict_data', methods=['POST'])
def predict_data():
    pclass = request.form['pclass']
    sex = request.form['sex']
    age = request.form['age']
    sibsp = request.form['sibsp']
    parch = request.form['parch']
    fare = request.form['fare']
    embarked = request.form['embarked']

    df = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

    output = preprocess_and_model(df)

    return render_template('output.html', data=df.values, columns=df.columns, output=output)


@app.route('/predict_file', methods=['POST'])
def predict_file():
    file = request.files['file']

    if file.filename.rsplit('.')[-1] not in ['csv', 'CSV']:
        return render_template('index.html', output='Invalid file. Allowed files [CSV]')

    df = pd.read_csv(file)
    output = preprocess_and_model(df)
    
    return render_template('output.html', data=df.values, columns=df.columns, output=output)


def preprocess_and_model(df):
    features_name = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

    df = df[features_name].copy()

    full_transformer = joblib.load('model/full_transformer.pickle')
    tree_clf = joblib.load('model/tree_clf.pickle')

    prepared_data = full_transformer.transform(df)
    prediction = tree_clf.predict(prepared_data)

    return prediction


if __name__ == '__main__':
    app.run(debug=True)

