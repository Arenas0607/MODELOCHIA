from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

modelo = joblib.load("modelo_trafico.pkl")
le_dia = joblib.load("encoder_dia.pkl")

@app.route('/')
def index():
    return render_template('index.html', resultado=None)

@app.route('/predecir', methods=['POST'])
def predecir():
    
    hora = int(request.form['hora'])
    dia_semana = request.form['dia_semana']  # verificar funcion genera 550 e
    dia_codificado = le_dia.transform([dia_semana])[0]
    prediccion = modelo.predict([[hora, dia_codificado]])
    return render_template('index.html', resultado=round(prediccion[0], 2))
if __name__ == '__main__':
    app.run(debug=True)

#verificar modelo 
