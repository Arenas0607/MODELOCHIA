from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

modelo = joblib.load('modelo_trafico.pkl')
le_dia = joblib.load('encoder_dia.pkl')

# _____________________________________________________
@app.route('/')
def home():
    return redirect(url_for('inicio'))


@app.route('/inicio', methods=['GET', 'POST'])
def inicio():
    resultado = request.args.get('resultado')
    error = request.args.get('error')
    return render_template('inicio.html', resultado=resultado, error=error)

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/objetivos')
def objetivos():
    return render_template('objetivos.html')

@app.route('/ingenieriaDatos')
def ingenieria_datos():
    return render_template('ingenieriaDatos.html')

@app.route('/ingenieriaModelo')
def ingenieria_modelo():
    return render_template('ingenieriaModelo.html')

@app.route('/despliegue')
def despliegue():
    return render_template('despliegue.html')

@app.route('/pdg')
def pdg():
    return render_template('pdg.html')

#__________________________________________________________-

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        hora = int(request.form.get('hora', 12))
        dia_semana = request.form.get('dia_semana', 'Lunes')
        dia_codificado = le_dia.transform([dia_semana])[0]
        prediccion = modelo.predict([[hora, dia_codificado]])
        resultado_prediccion = round(prediccion[0], 2)
        return redirect(url_for('index', resultado=resultado_prediccion))
    except Exception as e:
        error_mensaje = f"Error en la predicción: {str(e)}"
        return redirect(url_for('index', error=error_mensaje))

if __name__ == '__main__':
    app.run(debug=True)