from flask import Flask, request, jsonify, send_from_directory
import pickle
import os

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar el modelo entrenado y el vectorizador
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Ruta para la página principal
@app.route('/')
def home():
    return "Bienvenido a la API de Análisis de Sentimientos"

# Ruta para el favicon
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Ruta para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    try:
        data = request.json
        review = data['review']
        
        # Verificar si la reseña está presente
        if not review:
            return jsonify({'error': 'No review provided'}), 400
        
        # Transformar la entrada
        review_vec = vectorizer.transform([review])
        
        # Realizar predicción
        prediction = model.predict(review_vec)[0]
        
        # Devolver resultado
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Ejecutar la aplicación en modo de depuración
    app.run(debug=True, host='0.0.0.0')
