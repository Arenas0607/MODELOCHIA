# no olvidar realizar el cambio de dias numerios a alfabeticos :)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
data = {
    "hora": [7, 8, 9, 17, 18, 19, 10, 11],
    "dia": ["Lunes", "Lunes", "Lunes", "Lunes", "Lunes", "Lunes", "Martes", "Martes"],
    "trafico": [8, 9, 10, 9, 8, 7, 4, 3]
}
df = pd.DataFrame(data)
le_dia = LabelEncoder()
df["dia_codificada"] = le_dia.fit_transform(df["dia"])
X = df[["hora", "dia_codificada"]]
y = df["trafico"]
modelo = RandomForestRegressor()
modelo.fit(X, y)
#________________________________________________________________
joblib.dump(modelo, "modelo_trafico.pkl")
joblib.dump(le_dia, "encoder_dia.pkl")

print("Modelo y codificador entrenados y guardados.")  #verificar
