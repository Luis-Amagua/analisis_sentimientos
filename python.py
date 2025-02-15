from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.model_selection import train_test_split

# Usando el dataset de noticias para una demostración
data = fetch_20newsgroups_vectorized(subset='train')

# Dividiendo en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
