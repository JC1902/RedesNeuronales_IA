from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import prueba_main  # Importa tu script de OpenVino aquí

app = Flask(__name__)
CORS(app)  # Permite solicitudes CORS desde cualquier origen

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/ping', methods=['GET'])
def ping():
    print("Conectando..")
    return jsonify({'message': 'pong'}), 200

@app.route('/extract-text', methods=['POST'])
def extract_text():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Llama a tu función de OpenVino para extraer texto
    extracted_text = prueba_main.metodo_principal(filepath)
    
    # Elimina la imagen después de procesarla
    os.remove(filepath)
    print(extracted_text)
    return jsonify({'extracted_text': extracted_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
