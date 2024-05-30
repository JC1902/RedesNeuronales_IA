import requests

url = 'http://192.168.1.4:5000/extract-text'
files = {'image': open('./uploads/hola.jpg', 'rb')}
response = requests.post(url, files=files)

print(response.json())