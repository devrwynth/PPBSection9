import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YOLO App',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ImagePicker _picker = ImagePicker();
  YOLO? _yolo;
  XFile? _image;
  List<YOLOResult>? _results;
  bool _isProcessing = false;

  @override
  void initState() {
    super.initState();
    _initPredictor();
  }

  /// 4. Inference Logic: Initialize the YOLO predictor with the model path
  Future<void> _initPredictor() async {
    // FIX: Set useGpu to false to avoid "SPLIT" operation errors on devices with limited GPU support
    _yolo = YOLO(
      modelPath: 'assets/models/yolo11n_int8.tflite',
      useGpu: false,
    );
    await _yolo!.loadModel();
  }

  /// 6. Cleanup: Dispose of the predictor when the widget is destroyed
  @override
  void dispose() {
    _yolo?.dispose();
    super.dispose();
  }

  /// 3 & 4. Image Picking and Inference Logic
  Future<void> _pickImage() async {
    final XFile? pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _image = pickedFile;
        _results = null;
        _isProcessing = true;
      });

      try {
        final Uint8List imageBytes = await File(pickedFile.path).readAsBytes();
        
        // Run predict() on the picked image bytes
        final results = await _yolo?.predict(imageBytes);

        if (results != null && results.containsKey('detections')) {
          final List<dynamic> detectionsData = results['detections'];
          setState(() {
            _results = detectionsData
                .map((d) => YOLOResult.fromMap(d as Map<dynamic, dynamic>))
                .toList();
          });
        }
      } catch (e) {
        debugPrint("Error during prediction: $e");
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Inference error: $e')),
          );
        }
      } finally {
        setState(() {
          _isProcessing = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // 1. UI Layout: Use a Stack layout (No AppBar)
      body: Stack(
        children: [
          // 2. Initial State: Show text if no image is selected
          if (_image == null)
            const Center(
              child: Padding(
                padding: EdgeInsets.symmetric(horizontal: 32.0),
                child: Text(
                  'No image picked. Use the button below to start',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 16, color: Colors.grey),
                ),
              ),
            )
          else ...[
            // 5. Display State: Display the image fullscreen
            Positioned.fill(
              child: Image.file(
                File(_image!.path),
                fit: BoxFit.contain,
              ),
            ),
            // Overlay bounding boxes using CustomPainter
            if (_results != null)
              Positioned.fill(
                child: CustomPaint(
                  painter: DetectionPainter(_results!),
                ),
              ),
          ],

          if (_isProcessing)
            const Center(child: CircularProgressIndicator()),

          // 3 & 5. FloatingActionButton (Pick button remains visible on top)
          Positioned(
            bottom: 40,
            left: 0,
            right: 0,
            child: Center(
              child: FloatingActionButton.extended(
                onPressed: _isProcessing ? null : _pickImage,
                label: Text(_image == null ? 'Pick Image' : 'Pick New Image'),
                icon: const Icon(Icons.photo_library),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// CustomPainter to overlay bounding boxes on the image
class DetectionPainter extends CustomPainter {
  final List<YOLOResult> results;

  DetectionPainter(this.results);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = Colors.cyanAccent;

    final textPainter = TextPainter(
      textDirection: TextDirection.ltr,
    );

    for (final result in results) {
      // Use normalizedBox to draw on the canvas correctly relative to widget size
      final rect = Rect.fromLTRB(
        result.normalizedBox.left * size.width,
        result.normalizedBox.top * size.height,
        result.normalizedBox.right * size.width,
        result.normalizedBox.bottom * size.height,
      );

      // Draw bounding box
      canvas.drawRect(rect, paint);

      // Draw label and confidence
      textPainter.text = TextSpan(
        text: '${result.className} ${(result.confidence * 100).toStringAsFixed(0)}%',
        style: const TextStyle(
          color: Colors.black,
          backgroundColor: Colors.cyanAccent,
          fontSize: 14,
          fontWeight: FontWeight.bold,
        ),
      );
      textPainter.layout();
      textPainter.paint(canvas, Offset(rect.left, rect.top > 20 ? rect.top - 20 : rect.top));
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
