import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
data = pd.DataFrame({
    'hora': range(24),
    'dia_semana': [i % 7 for i in range(24)],
    'trafico': [min((i % 12) + (i // 3), 10) for i in range(24)]
})

X = data[['hora', 'dia_semana']]
y = data['trafico']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

joblib.dump(modelo, 'modelo_trafico.pkl')
#corregido