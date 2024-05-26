import os
import requests
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
from PIL import Image
import matplotlib.pyplot as plt

# Cargar funciones de utilidad
def load_image( image_path ):
    if image_path.startswith( "http" ):
        response = requests.get( image_path, stream=True )
        img = Image.open( response.raw )
    else:
        img = Image.open( image_path )

    return np.array( img )

def multiply_by_ratio( ratio_x, ratio_y, box ):
    return [ max( shape * ratio_y, 10 ) if idx % 2 else shape * ratio_x for idx, shape in enumerate( box[ :-1 ] ) ]

def run_preprocesing_on_crop( crop, net_shape ):
    temp_img = cv2.resize( crop, net_shape )
    temp_img = temp_img.reshape( ( 1, ) * 2 + temp_img.shape )

    return temp_img

def convert_result_to_image( bgr_image, resized_image, boxes, annotations, threshold = 0.3, conf_labels = True ):
    colors = { "red": ( 255, 0, 0 ), "green": ( 0, 255, 0 ), "white": ( 255, 255, 255 ) }
    ( real_y, real_x ), ( resized_y, resized_x ) = bgr_image.shape[ :2 ], resized_image.shape[ :2 ]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

    rgb_image = cv2.cvtColor( bgr_image, cv2.COLOR_BGR2RGB )

    for box, annotation in zip( boxes, annotations ):
        conf = box[ -1 ]

        if conf > threshold:
            ( x_min, y_min, x_max, y_max ) = map( int, multiply_by_ratio( ratio_x, ratio_y, box ) )
            cv2.rectangle( rgb_image, ( x_min, y_min ), ( x_max, y_max ), colors[ "green" ], 3 )

            if conf_labels:
                ( text_w, text_h ), _ = cv2.getTextSize( f"{ annotation }", cv2.FONT_HERSHEY_TRIPLEX, 0.8, 1 )
                image_copy = rgb_image.copy()

                cv2.rectangle( 
                    image_copy,
                    ( x_min, y_min - text_h - 10 ),
                    ( x_min + text_w, y_min - 10 ),
                    colors[ "white" ],
                    -1,
                )

                cv2.addWeighted( image_copy, 0.4, rgb_image, 0.6, 0, rgb_image )

                cv2.putText(
                    rgb_image,
                    f"{annotation}",
                    ( x_min, y_min -10 ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    colors[ "red" ],
                    1,
                    cv2.LINE_AA,
                )
    return rgb_image

# Inicializamos OpenVINO
core = ov.Core()

# Directorio de los modelos
model_dir = Path( "model" )
precision = "FP16"
detection_model = "horizontal-text-detection-0001"
recognition_model = "text-recognition-resnet-fc"

# Descargar y convertir los modelos en caso de que no existan
if not model_dir.exists():
    model_dir.mkdir(exist_ok=True)
    download_command = f"omz_downloader --name {detection_model},{recognition_model} --output_dir {model_dir} --cache_dir {model_dir} --precision {precision} --num_attempts 5"

    os.system(download_command)

    convert_command = f"omz_converter --name {recognition_model} --precisions {precision} --download_dir {model_dir} --output_dir {model_dir}"

    os.system(convert_command)

detection_model_path = ( model_dir/"intel/horizontal-text-detection-0001"/precision/detection_model ).with_suffix(".xml")
recognition_model_path = ( model_dir/"public/text-recognition-resnet-fc"/precision/recognition_model ).with_suffix(".xml")

# Seleccionar el dispositivo
device = "CPU"

# Leer y compilar los modelos
detection_model = core.read_model( model=detection_model_path, weights=detection_model_path.with_suffix(".bin") )
detection_compiled_model = core.compile_model( model=detection_model, device_name=device )

recognition_model = core.read_model( model=recognition_model_path, weights=recognition_model_path.with_suffix(".bin") )
recognition_compiled_model = core.compile_model( model=recognition_model, device_name=device )

detetcion_input_layer = detection_compiled_model.input( 0 )
recognition_output_layer = recognition_compiled_model.output( 0 )
recognition_input_layer = recognition_compiled_model.input( 0 )

# Parámetros del modelo de detección
N, C, H, W = detetcion_input_layer.shape

# Cargar una imagen de ejemplo
image_file = "https://www.mayoreototal.mx/cdn/shop/products/7501020515343_1000x.jpg?v=1626803116"
image = load_image( image_file )

# Procesar la imagen con el modelo
resized_image = cv2.resize( image, ( W, H ) )
input_image = np.expand_dims( resized_image.transpose( 2, 0, 1 ), 0 )

# Realizar la detección
output_key = detection_compiled_model.output( "boxes" )
boxes = detection_compiled_model( [ input_image ] )[ output_key ]
boxes = boxes[ ~np.all( boxes == 0, axis=1 ) ]

# Parámetros del modelo de reconocimiento
_, _, H, W = recognition_input_layer.shape
( real_y, real_x ), ( resized_y, resized_x ) = image.shape[ :2 ], resized_image.shape[ :2 ]
ratio_x, ratio_y = real_x / resized_x, real_y / resized_y
grayscale_image = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
letters = "~0123456789abcdefghijkmnopqrstuwxyz"

annotations = []
cropped_images = []

for crop in boxes:
    ( x_min, y_min, x_max, y_max ) = map( int, multiply_by_ratio( ratio_x, ratio_y, crop ) )
    image_crop = run_preprocesing_on_crop( grayscale_image[ y_min:y_max, x_min:x_max ], ( W, H ) )
    result = recognition_compiled_model( [ image_crop ] )[ recognition_output_layer ]
    recognition_results_test = np.squeeze( result )

    annotation = []

    for letter in recognition_results_test:
        parsed_letter = letters[ letter.argmax() ]

        if parsed_letter == letters[ 0 ]:
            break

        annotation.append( parsed_letter )
    annotations.append( "".join( annotation ) )
    cropped_images.append( image[ y_min:y_max, x_min:x_max ] )

# Mostrar los resultados
output_image = convert_result_to_image( image, resized_image, boxes, annotations, conf_labels=True )

plt.figure( figsize=( 12, 12 ) )
plt.imshow( output_image )
plt.show()

for cropped_image, annotation in zip( cropped_images, annotations ):
    plt.imshow( cropped_image, cmap='gray' )
    plt.title( "".join( annotation ) )
    plt.show()

print( [ annotation for _, annotation in sorted( zip( boxes, annotations ), key=lambda x: x[ 0 ][ 0 ] ** 2 + x[ 0 ][ 1 ] ** 2 ) ] )