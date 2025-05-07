from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
modelo = joblib.load('modelo_trafico.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    hora = int(request.form['hora'])
    dia_semana = int(request.form['dia_semana'])
    prediccion = modelo.predict(np.array([[hora, dia_semana]]))[0]
    return render_template('index.html', resultado=round(prediccion, 2))

if __name__ == '__main__':
    app.run(debug=True)
#verificar modelo 
