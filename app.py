from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

# Path ke file model TFLite
model_path = 'Fast-Food-Classification-BiT.tflite'

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Daftar nama kelas
class_names = ['Baked Potato', 'Burger', 'Crispy Chicken', 'Donut', 'Fries', 'Hot Dog', 'Pizza', 'Sandwich', 'Taco',
               'Taquito']

# Load data nutrisi dari file JSON
with open('fastfood.json', 'r') as file:
    fastfood_data = json.load(file)

# Fungsi untuk melakukan preprocessing gambar dengan tipe FLOAT32
def preprocess_image(image_data, target_size=(256, 256)):
    # Load gambar menggunakan PIL
    image = Image.open(image_data)

    # Konversi gambar ke mode RGB (jika bukan format RGB)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize gambar ke ukuran yang diinginkan
    image = image.resize(target_size)

    # Konversi nilai piksel menjadi FLOAT32
    image_array = np.array(image, dtype=np.float32) / 255.0

    # Ekspansi dimensi untuk membuat batch
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

# Fungsi untuk melakukan prediksi berdasarkan gambar menggunakan model TFLite
def predict_image(image_data):
    # Preprocessing gambar
    processed_image = preprocess_image(image_data)

    # Salin data gambar ke tensor input model TFLite
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], processed_image)

    # Lakukan inferensi
    interpreter.invoke()

    # Dapatkan output dari tensor output model TFLite
    output_details = interpreter.get_output_details()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Proses hasil output
    predicted_class = class_names[np.argmax(output)]
    confidence = np.max(output)

    # Cari nutrisi sesuai dengan kelas yang diprediksi
    nutrition = {}
    for food in fastfood_data:
        if food['Makanan'] == predicted_class:
            nutrition = {
                'Kalori': food['Kalori'],
                'Karbohidrat': food['Karbohidrat'],
                'Lemak': food['Lemak'],
                'Protein': food['Protein'],
                'Tautan': food['Tautan']
            }
            break

    return predicted_class, confidence, nutrition

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    try:
        predicted_class, confidence, nutrition = predict_image(file)
    except Exception as e:
        return jsonify({'error': str(e)})

    # Atur urutan respons sesuai keinginan di dalam fungsi predict
    response = {
        'prediksi': predicted_class,
        'tingkat_kepercayaan': float(confidence),
    }

    nutrition_response = {
            'nutrisi': nutrition
        }

    return jsonify(response, nutrition_response)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
