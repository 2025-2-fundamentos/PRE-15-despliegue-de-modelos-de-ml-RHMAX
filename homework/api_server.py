## api_server.py


"""API server example"""

#
# Usage from command line:
# curl http://127.0.0.1:5000 -X POST -H "Content-Type: application/json" \
# -d '{"bathrooms": "2", "bedrooms": "3", "sqft_living": "1800", \
# "sqft_lot": "2200", "floors": "1", "waterfront": "1", "condition": "3"}'
#

# Windows:
# curl http://127.0.0.1:5000 -X POST -H "Content-Type: application/json" -d "{\"bathrooms\": \"2\", \"bedrooms\": \"3\", \"sqft_living\": \"1800\", \"sqft_lot\": \"2200\", \"floors\": \"1\", \"waterfront\": \"1\", \"condition\": \"3\"}"

import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
    # Obtener datos del request
    data = request.get_json()
    
    # Filtrar solo las características necesarias
    expected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                        'floors', 'waterfront', 'view', 'condition', 'grade', 
                        'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']
    
    filt_args = {key: [data[key]] for key in expected_features if key in data}
    df = pd.DataFrame.from_dict(filt_args)

    # Cargar el modelo
    with open("homework/house_predictor.pkl", "rb") as file:
        loaded_content = pickle.load(file)
        
        # Si el contenido es un diccionario, extraer el modelo
        if isinstance(loaded_content, dict):
            # Buscar el modelo en el diccionario
            if 'model' in loaded_content:
                model = loaded_content['model']
            elif 'estimator' in loaded_content:
                model = loaded_content['estimator']
            else:
                # Tomar el primer valor que sea un modelo
                for key, value in loaded_content.items():
                    if hasattr(value, 'predict'):
                        model = value
                        break
                else:
                    return jsonify({'error': 'No se encontró un modelo válido en el archivo pickle'}), 500
        else:
            model = loaded_content

    # Hacer la predicción
    try:
        prediction = model.predict(df)
        return str(prediction[0])
    except Exception as e:
        return jsonify({'error': f'Error en predicción: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
    