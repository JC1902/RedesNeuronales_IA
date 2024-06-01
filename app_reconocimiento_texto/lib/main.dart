import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import 'dart:io'; // Importar para usar File
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(MyApp());
}

class HistoryPage extends StatefulWidget {
  const HistoryPage({super.key});

  @override
  _HistoryPageState createState() => _HistoryPageState();
}

class _HistoryPageState extends State<HistoryPage> {
  Future<List<dynamic>> _fetchData() async {
    final response = await http.get(Uri.parse('https://20130763.000webhostapp.com/historialDatosRegistro.php'));

    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception('Failed to load data');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Historial de Entradas'),
      ),
      body: FutureBuilder<List<dynamic>>(
        future: _fetchData(),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          } else {
            final data = snapshot.data!;
            return ListView.builder(
              itemCount: data.length,
              itemBuilder: (context, index) {
                final item = data[index];
                return ListTile(
                  title: Text(item['descripcion']),
                  subtitle: Text('${item['fecha']} ${item['hora']}'),
                );
              },
            );
          }
        },
      ),
    );
  }
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
        debugShowCheckedModeBanner: false,
        title: 'Detector de Texto AI',
        theme: ThemeData(
          primarySwatch: Colors.blue,
          appBarTheme: const AppBarTheme(
            backgroundColor: Color(0xFF003366), // Azul rey
            foregroundColor: Colors.white, // Texto en blanco
          ),
          elevatedButtonTheme: ElevatedButtonThemeData(
            style: ButtonStyle(
              backgroundColor: MaterialStateProperty.all(Color(0xFF003366)), // Azul rey
              foregroundColor: MaterialStateProperty.all(Colors.white), // Texto en blanco
            ),
          ),
        ),
        home: SplashScreen(),
      ),
    );
  }
}

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    Future.delayed(const Duration(seconds: 3), () {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => MyHomePage()),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text(
              'TextFinder AI',
              style: TextStyle(
                fontSize: 36.0,
                fontWeight: FontWeight.bold,
                fontFamily: 'Retro Gaming',
              ),
            ),
            const SizedBox(height: 40.0), // Aumenta el espacio entre el texto y la imagen
            SizedBox(
              height: 300.0, // Ajusta el tamaño de la caja de imagen
              width: 400.0, // Ajusta el tamaño de la caja de imagen
              child: Image.asset('assets/logo.png'), // Cambia 'assets/icon.png' por la ruta de tu icono
            ),
            const SizedBox(height: 40.0), // Aumenta el espacio entre la imagen y el progress bar
            const SizedBox(
              height: 60.0, // Ajusta el tamaño del CircularProgressIndicator
              width: 60.0, // Ajusta el tamaño del CircularProgressIndicator
              child: CircularProgressIndicator(), // Progress bar circular
            ),
          ],
        ),
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
    final uri = Uri.parse('http://192.168.1.14:5000/ping');
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
        title: const Text('Menú Principal'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
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
                          child: const Text('Cerrar'),
                        ),
                      ],
                    ),
                  );
                }
              },
              child: const Text('Abrir la Cámara'),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => HistoryPage()),
                );
              },
              child: const Text('Textos Identificados'),
            ),
          ],
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

    final uri = Uri.parse('http://192.168.1.14:5000/extract-text');
    final request = http.MultipartRequest('POST', uri)
      ..files.add(await http.MultipartFile.fromPath('image', image.path));

    try {
      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final responseData = response.body;
        final decodedData = json.decode(responseData);
        final extractedTexts = List<String>.from(decodedData['extracted_text']);
        final concatenatedText = extractedTexts.join('-');

        showDialog(
          context: context,
          builder: (context) => AlertDialog(
            title: const Text('Textos Extraídos'),
            content: SingleChildScrollView(
              child: ListBody(
                children: extractedTexts.map((text) => Text(text)).toList(),
              ),
            ),
            actions: [
              TextButton(
                onPressed: () {
                  Navigator.of(context).pop();
                },
                child: const Text('Cerrar'),
              ),
              TextButton(
                onPressed: () async {
                  await _sendDataToServer(context, concatenatedText);
                  Navigator.of(context).popUntil((route) => route.isFirst);
                },
                child: const Text('Enviar datos'),
              ),
            ],
          ),
        );
      } else {
        print('Fallo al extraer los textos, el servidor a respondido con el código: ${response.statusCode}');
        throw Exception('Failed to extract text');
      }
    } catch (e) {
      print('Error extracting text: $e');
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Error'),
          content: Text('Fallo al extraer el texto: $e'),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
              },
              child: const Text('Cerrar'),
            ),
          ],
        ),
      );
    }
  }

  Future<void> _sendDataToServer(BuildContext context, String concatenatedText) async {
    final uri = Uri.parse('https://20130763.000webhostapp.com/postTablaRegistros.php');
    final response = await http.post(
      uri,
      headers: {
        'Content-Type': 'application/json',
      },
      body: json.encode({'extracted_text': concatenatedText}),
    );

    if (response.statusCode != 200) {
      throw Exception('Failed to send data to server');
    }
  }

  @override
  Widget build(BuildContext context) {
    final image = Provider.of<ImageProviderModel>(context).image;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Imagen Tomada'),
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
