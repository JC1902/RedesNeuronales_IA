import os
from pathlib import Path
import numpy as np
import cv2
import openvino as ov
from PIL import Image
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

# Inicializamos Flask
app = Flask(__name__)

# Inicializamos OpenVINO
core = ov.Core()

# Directorio de los modelos
model_dir = Path("model")
precision = "FP16"
detection_model = "horizontal-text-detection-0001"
recognition_model = "text-recognition-resnet-fc"

# Rutas de los modelos
detection_model_path = (model_dir / "intel/horizontal-text-detection-0001" / precision / detection_model).with_suffix(".xml")
recognition_model_path = (model_dir / "public/text-recognition-resnet-fc" / precision / recognition_model).with_suffix(".xml")

# Leer y compilar los modelos
detection_model = core.read_model(model=detection_model_path, weights=detection_model_path.with_suffix(".bin"))
detection_compiled_model = core.compile_model(model=detection_model, device_name="CPU")

recognition_model = core.read_model(model=recognition_model_path, weights=recognition_model_path.with_suffix(".bin"))
recognition_compiled_model = core.compile_model(model=recognition_model, device_name="CPU")

# Funciones de utilidad
def multiply_by_ratio(ratio_x, ratio_y, box):
    return [max(shape * ratio_y, 10) if idx % 2 else shape * ratio_x for idx, shape in enumerate(box[:-1])]

def run_preprocesing_on_crop(crop, net_shape):
    temp_img = cv2.resize(crop, net_shape)
    temp_img = temp_img.reshape((1,) * 2 + temp_img.shape)
    return temp_img

def convert_result_to_image(bgr_image, resized_image, boxes, annotations, threshold=0.3, conf_labels=True):
    colors = {"red": (255, 0, 0), "green": (0, 255, 0), "white": (255, 255, 255)}
    (real_y, real_x), (resized_y, resized_x) = bgr_image.shape[:2], resized_image.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    for box, annotation in zip(boxes, annotations):
        conf = box[-1]

        if conf > threshold:
            (x_min, y_min, x_max, y_max) = map(int, multiply_by_ratio(ratio_x, ratio_y, box))
            cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), colors["green"], 3)

            if conf_labels:
                (text_w, text_h), _ = cv2.getTextSize(f"{annotation}", cv2.FONT_HERSHEY_TRIPLEX, 0.8, 1)
                image_copy = rgb_image.copy()

                cv2.rectangle(
                    image_copy,
                    (x_min, y_min - text_h - 10),
                    (x_min + text_w, y_min - 10),
                    colors["white"],
                    -1,
                )

                cv2.addWeighted(image_copy, 0.4, rgb_image, 0.6, 0, rgb_image)

                cv2.putText(
                    rgb_image,
                    f"{annotation}",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    colors["red"],
                    1,
                    cv2.LINE_AA,
                )
    return rgb_image

# Ruta para recibir y procesar la imagen
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"})
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        image = np.array(Image.open(file.stream))

        # Procesar la imagen con el modelo
        resized_image = cv2.resize(image, (W, H))
        input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)

        # Realizar la detección
        output_key = detection_compiled_model.output("boxes")
        boxes = detection_compiled_model([input_image])[output_key]
        boxes = boxes[~np.all(boxes == 0, axis=1)]

        # Parámetros del modelo de reconocimiento
        _, _, H, W = recognition_input_layer.shape
        (real_y, real_x), (resized_y, resized_x) = image.shape[:2], resized_image.shape[:2]
        ratio_x, ratio_y = real_x / resized_x, real_y / resized_y
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        letters = "~0123456789abcdefghijkmnopqrstuwxyz"

        annotations = []
        cropped_images = []

        for crop in boxes:
            (x_min, y_min, x_max, y_max) = map(int, multiply_by_ratio(ratio_x, ratio_y, crop))
            image_crop = run_preprocesing_on_crop(grayscale_image[y_min:y_max, x_min:x_max], (W, H))
            result = recognition_compiled_model([image_crop])[recognition_output_layer]
            recognition_results_test = np.squeeze(result)

            annotation = []

            for letter in recognition_results_test:
                parsed_letter = letters[letter.argmax()]

                if parsed_letter == letters[0]:
                    break

                annotation.append(parsed_letter)
            annotations.append("".join(annotation))
            cropped_images.append(image[y_min:y_max, x_min:x_max])

        # Mostrar los resultados
        output_image = convert_result_to_image(image, resized_image, boxes, annotations, conf_labels=True)
        
        # Guardar la imagen de salida temporalmente
        output_image_path = 'output_image.jpg'
        Image.fromarray(output_image).save(output_image_path)

        # Devolver resultados
        return jsonify({"annotations": annotations, "output_image_url": request.host_url + output_image_path})

# Ejecutar la aplicación Flask
if __name__ == "__main__":
    app.run(debug=True)
