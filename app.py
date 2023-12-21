from flask import Flask, render_template, request
from Models import testing
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.preprocessing.text import Tokenizer

application = Flask(__name__)

@application.route('/', methods=['GET', 'POST'])
def index():
    model_path = 'static/account_model.sav'

    tokenizer_path = 'static/tokenizer.sav'

    if request.method == 'POST':
        r = str(request.form['input'])
        output = testing(r, model_path, tokenizer_path)
        return render_template('form.html', model=output)
    else:
        return render_template('form.html', model='')

if __name__ == '__main__':
    application.run(debug=True)