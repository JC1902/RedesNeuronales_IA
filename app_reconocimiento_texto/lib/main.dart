import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import 'dart:io'; // Importar para usar File

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
        title: 'Flutter Camera App',
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Flutter Camera App'),
      ),
      body: Center(
        child: ElevatedButton(
          onPressed: () => _openCamera(context),
          child: const Text('Open Camera'),
        ),
      ),
    );
  }
}

class DisplayPictureScreen extends StatelessWidget {
  const DisplayPictureScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final image = Provider.of<ImageProviderModel>(context).image;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Display Picture'),
      ),
      body: Center(
        child: image == null
            ? const Text('No image selected.')
            : Image.file(File(image.path)), // Usar File del paquete dart:io
      ),
    );
  }
}
