import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib


df = pd.read_csv("datos_trafico.csv") #verificar conexion de datos.
le_dia = LabelEncoder()
df["dia_codificada"] = le_dia.fit_transform(df["dia"])
X = df[["hora", "dia_codificada"]]
y = df["trafico"]
modelo = RandomForestRegressor()
modelo.fit(X, y)
joblib.dump(modelo, "modelo_trafico.pkl")
joblib.dump(le_dia, "encoder_dia.pkl")

print("Modelo entrenado y guardado correctamente :3.")

