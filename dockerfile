# Usamos una imagen base de Python
FROM python:3.9-slim

# Instalar las dependencias
RUN pip install --upgrade pip
RUN pip install scikit-learn pandas numpy matplotlib seaborn flask tensorflow_datasets

# Copiar tu código dentro del contenedor
COPY . /app

# Establecer el directorio de trabajo
WORKDIR /app

# Exponer el puerto en el que correrá la API
EXPOSE 5000

# Comando para correr tu aplicación
CMD ["python", "app.py"]
