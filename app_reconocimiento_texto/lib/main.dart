import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import 'dart:io'; // Importar para usar File
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(MyApp());
}

class ImageProviderModel with ChangeNotifier {
  XFile? _image;

  XFile? get image => _image;

  void setImage(XFile? image) {
    _image = image;
    notifyListeners();
  }
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (context) => ImageProviderModel(),
      child: MaterialApp(
        title: 'Detector de texto App',
        theme: ThemeData(
          primarySwatch: Colors.blue,
        ),
        home: MyHomePage(),
      ),
    );
  }
}

class MyHomePage extends StatelessWidget {
  final ImagePicker _picker = ImagePicker();

  MyHomePage({super.key});

  Future<void> _openCamera(BuildContext context) async {
    final XFile? image = await _picker.pickImage(source: ImageSource.camera);
    if (!context.mounted) return;
    if (image != null) {
      Provider.of<ImageProviderModel>(context, listen: false).setImage(image);
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => DisplayPictureScreen()),
      );
    }
  }

  Future<bool> _checkServerConnection() async {
    final uri = Uri.parse('http://192.168.1.4:5000/ping');
    try {
      print('Error connecting to server: conectando..');
      final response = await http.get(uri);
      if (response.statusCode == 200) {
        return true;
      }
    } catch (e) {
      print('Error connecting to server: $e');
    }
    return false;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Flutter Camera App'),
      ),
      body: Center(
        child: ElevatedButton(
          onPressed: () async {
            final connected = await _checkServerConnection();
            if (connected) {
              _openCamera(context);
            } else {
              showDialog(
                context: context,
                builder: (context) => AlertDialog(
                  title: const Text('Error'),
                  content: const Text('No se pudo conectar con el servidor.'),
                  actions: [
                    TextButton(
                      onPressed: () {
                        Navigator.of(context).pop();
                      },
                      child: const Text('Close'),
                    ),
                  ],
                ),
              );
            }
          },
          child: const Text('Open Camera'),
        ),
      ),
    );
  }
}

class DisplayPictureScreen extends StatelessWidget {
  const DisplayPictureScreen({super.key});

  Future<void> _extractText(BuildContext context) async {
    final image = Provider.of<ImageProviderModel>(context, listen: false).image;
    if (image == null) return;

    final uri = Uri.parse('http://192.168.1.4:5000/extract-text');
    final request = http.MultipartRequest('POST', uri)
      ..files.add(await http.MultipartFile.fromPath('image', image.path));

    try {
      final response = await request.send();
      if (response.statusCode == 200) {
        final responseData = await response.stream.bytesToString();
        final decodedData = json.decode(responseData);
        final extractedTexts = List<String>.from(decodedData['extracted_texts']);

        showDialog(
          context: context,
          builder: (context) => AlertDialog(
            title: const Text('Extracted Texts'),
            content: SingleChildScrollView(
              child: ListView.builder(
                shrinkWrap: true,
                itemCount: extractedTexts.length,
                itemBuilder: (context, index) {
                  return ListTile(
                    title: Text(extractedTexts[index]),
                  );
                },
              ),
            ),
            actions: [
              TextButton(
                onPressed: () {
                  Navigator.of(context).pop();
                },
                child: const Text('Close'),
              ),
            ],
          ),
        );
      } else {
        throw Exception('Failed to extract text');
      }
    } catch (e) {
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Error'),
          content: Text('Failed to extract text: $e'),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
              },
              child: const Text('Close'),
            ),
          ],
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final image = Provider.of<ImageProviderModel>(context).image;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Display Picture'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            image == null
                ? const Text('No image selected.')
                : Image.file(File(image.path)), // Usar File del paquete dart:io
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () => _extractText(context),
              child: const Text('Extraer texto'),
            ),
          ],
        ),
      ),
    );
  }
}
