# Instrucciones
## Crear entorno virtual
python -m venv .venv

## Activar el entorno
.venv/Scripts/Activate.ps1

## Comando para instalar los requisitos desde el txt
pip install -r requerimientos.txt

### En caso de haber algun error con los modelos ( carpeta que no inclui ) usa estos comando y corre el codigo
omz_downloader --name text-recognition-resnet-fc --output_dir model --precision FP16 --num_attempts 5
omz_converter --name text-recognition-resnet-fc --precisions FP16 --download_dir model --output_dir model

### Con eso deberia funcionar el código dentro de prueba_main
