import 'dart:io';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'firebase_options.dart';
import 'services.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);
  runApp(const ScriptScribeApp());
}

class ScriptScribeApp extends StatelessWidget {
  const ScriptScribeApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'ScriptScribe',
        theme: ThemeData(
          colorSchemeSeed: Colors.deepOrange,
          useMaterial3: true,
        ),
        home: const HomeScreen(),
      );
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _service = ScriptScribeService();
  final _history = HistoryService();
  final _picker = ImagePicker();

  File? _image;
  ScanResult? _result;
  bool _busy = false;
  String? _error;

  bool get _supported => !kIsWeb && (Platform.isAndroid || Platform.isIOS);

  @override
  void dispose() {
    _service.dispose();
    super.dispose();
  }

  Future<void> _pick(ImageSource source) async {
    final picked = await _picker.pickImage(source: source);
    if (picked == null) return;
    setState(() {
      _image = File(picked.path);
      _result = null;
      _error = null;
      _busy = true;
    });
    try {
      final result = await _service.process(_image!);
      setState(() => _result = result);
      if (result.extracted.isNotEmpty) {
        // History is best-effort; don't fail the scan if Firestore is unreachable.
        _history.save(result).catchError((e) => debugPrint('save failed: $e'));
      }
    } catch (e) {
      setState(() => _error = '$e');
    } finally {
      setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ScriptScribe'),
        actions: [
          IconButton(
            icon: const Icon(Icons.history),
            tooltip: 'History',
            onPressed: () => Navigator.push(context,
                MaterialPageRoute(builder: (_) => HistoryScreen(_history))),
          ),
        ],
      ),
      body: !_supported
          ? const Center(
              child: Padding(
                padding: EdgeInsets.all(24),
                child: Text(
                    'On-device OCR runs on Android and iOS only. Please use the mobile app.'),
              ),
            )
          : ListView(
              padding: const EdgeInsets.all(16),
              children: [
                Row(
                  children: [
                    Expanded(
                      child: FilledButton.icon(
                        onPressed:
                            _busy ? null : () => _pick(ImageSource.gallery),
                        icon: const Icon(Icons.photo_library),
                        label: const Text('Gallery'),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: FilledButton.tonalIcon(
                        onPressed:
                            _busy ? null : () => _pick(ImageSource.camera),
                        icon: const Icon(Icons.camera_alt),
                        label: const Text('Camera'),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                if (_image != null) Image.file(_image!, height: 220),
                if (_busy)
                  const Padding(
                    padding: EdgeInsets.all(24),
                    child: Center(child: CircularProgressIndicator()),
                  ),
                if (_error != null)
                  Text(_error!,
                      style: TextStyle(
                          color: Theme.of(context).colorScheme.error)),
                if (_result != null) ResultView(_result!),
              ],
            ),
    );
  }
}

class ResultView extends StatelessWidget {
  final ScanResult result;
  const ResultView(this.result, {super.key});

  @override
  Widget build(BuildContext context) {
    if (result.extracted.isEmpty) {
      return const Padding(
        padding: EdgeInsets.all(16),
        child: Text('No text detected.'),
      );
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _section(context, 'Extracted text', result.extracted),
        _section(context, 'Transliteration (IAST)', result.transliterated),
        _section(context, 'English translation', result.translated),
      ],
    );
  }

  Widget _section(BuildContext context, String title, String body) => Card(
        margin: const EdgeInsets.only(top: 12),
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: Theme.of(context).textTheme.titleSmall),
              const SizedBox(height: 6),
              SelectableText(body, style: const TextStyle(fontSize: 16)),
            ],
          ),
        ),
      );
}

class HistoryScreen extends StatelessWidget {
  final HistoryService history;
  const HistoryScreen(this.history, {super.key});

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('History')),
        body: FutureBuilder<Stream<QuerySnapshot<Map<String, dynamic>>>>(
          future: history.stream(),
          builder: (context, streamSnap) {
            if (streamSnap.hasError) {
              return Center(child: Text('${streamSnap.error}'));
            }
            if (!streamSnap.hasData) {
              return const Center(child: CircularProgressIndicator());
            }
            return StreamBuilder<QuerySnapshot<Map<String, dynamic>>>(
              stream: streamSnap.data,
              builder: (context, snap) {
                if (snap.hasError) return Center(child: Text('${snap.error}'));
                final docs = snap.data?.docs ?? [];
                if (docs.isEmpty) {
                  return const Center(child: Text('No scans yet.'));
                }
                return ListView(
                  children: [
                    for (final d in docs)
                      ListTile(
                        title: Text(d['extracted'] ?? '',
                            maxLines: 1, overflow: TextOverflow.ellipsis),
                        subtitle: Text(d['translated'] ?? '',
                            maxLines: 2, overflow: TextOverflow.ellipsis),
                        onTap: () => Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (_) => Scaffold(
                              appBar: AppBar(title: const Text('Scan')),
                              body: ListView(
                                padding: const EdgeInsets.all(16),
                                children: [
                                  ResultView(ScanResult(
                                      d['extracted'] ?? '',
                                      d['transliterated'] ?? '',
                                      d['translated'] ?? '')),
                                ],
                              ),
                            ),
                          ),
                        ),
                      ),
                  ],
                );
              },
            );
          },
        ),
      );
}
