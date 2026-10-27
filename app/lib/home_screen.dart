import 'dart:io';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

import 'services.dart';
import 'theme.dart';

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
  String _stage = '';
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
      _stage = 'Preparing…';
    });
    try {
      final result = await _service.process(_image!,
          onStage: (s) => mounted ? setState(() => _stage = s) : null);
      if (!mounted) return;
      setState(() => _result = result);
      if (result.extracted.isNotEmpty) {
        // History is best-effort; don't fail the scan if Firestore is unreachable.
        _history.save(result).catchError((e) => debugPrint('save failed: $e'));
      }
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ScriptScribe'),
        actions: [
          IconButton(
            icon: const Icon(Icons.history_rounded),
            tooltip: 'History',
            onPressed: () => Navigator.push(context,
                MaterialPageRoute(builder: (_) => HistoryScreen(_history))),
          ),
        ],
      ),
      body: !_supported
          ? const _Unsupported()
          : SafeArea(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(20, 8, 20, 32),
                children: [
                  _PreviewCard(image: _image, busy: _busy, stage: _stage),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Expanded(
                        child: FilledButton.icon(
                          onPressed:
                              _busy ? null : () => _pick(ImageSource.camera),
                          icon: const Icon(Icons.camera_alt_rounded),
                          label: const Text('Camera'),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed:
                              _busy ? null : () => _pick(ImageSource.gallery),
                          icon: const Icon(Icons.photo_library_rounded),
                          label: const Text('Gallery'),
                        ),
                      ),
                    ],
                  ),
                  if (_error != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 16),
                      child: Text(_error!,
                          style: TextStyle(
                              color: Theme.of(context).colorScheme.error)),
                    ),
                  if (_result != null && !_busy) ...[
                    const SizedBox(height: 8),
                    ResultView(_result!),
                  ],
                ],
              ),
            ),
    );
  }
}

class _Unsupported extends StatelessWidget {
  const _Unsupported();

  @override
  Widget build(BuildContext context) => const Center(
        child: Padding(
          padding: EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              AppLogo(height: 120),
              SizedBox(height: 24),
              Text(
                'On-device OCR runs on Android and iOS only.\nPlease open the mobile app.',
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      );
}

class _PreviewCard extends StatelessWidget {
  final File? image;
  final bool busy;
  final String stage;
  const _PreviewCard(
      {required this.image, required this.busy, required this.stage});

  @override
  Widget build(BuildContext context) {
    return AspectRatio(
      aspectRatio: 4 / 3,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(24),
        child: Container(
          decoration: BoxDecoration(
            color: Colors.white,
            border: Border.all(color: kTeal.withValues(alpha: 0.12)),
            borderRadius: BorderRadius.circular(24),
          ),
          child: Stack(
            fit: StackFit.expand,
            children: [
              if (image == null)
                Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.add_photo_alternate_rounded,
                        size: 56, color: kTeal.withValues(alpha: 0.35)),
                    const SizedBox(height: 12),
                    Text('Capture or choose an image to begin',
                        style:
                            TextStyle(color: Colors.black.withValues(alpha: 0.5))),
                  ],
                )
              else
                Image.file(image!, fit: BoxFit.cover),
              AnimatedSwitcher(
                duration: const Duration(milliseconds: 250),
                child: busy
                    ? Container(
                        key: const ValueKey('busy'),
                        color: kTeal.withValues(alpha: 0.75),
                        padding: const EdgeInsets.all(24),
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const CircularProgressIndicator(
                                color: Colors.white),
                            const SizedBox(height: 20),
                            Text(stage,
                                textAlign: TextAlign.center,
                                style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 16,
                                    fontWeight: FontWeight.w600)),
                          ],
                        ),
                      )
                    : const SizedBox.shrink(key: ValueKey('idle')),
              ),
            ],
          ),
        ),
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
        padding: EdgeInsets.all(24),
        child: Center(child: Text('No text detected. Try a clearer image.')),
      );
    }
    return Column(
      children: [
        _section(context, Icons.text_fields_rounded, 'Extracted text',
            result.extracted, kTeal),
        _section(context, Icons.abc_rounded, 'Transliteration (IAST)',
            result.transliterated, kOrange),
        _section(context, Icons.translate_rounded, 'English translation',
            result.translated, kTeal),
      ],
    );
  }

  Widget _section(BuildContext context, IconData icon, String title,
          String body, Color accent) =>
      Padding(
        padding: const EdgeInsets.only(top: 14),
        child: Card(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 14, 8, 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(icon, color: accent, size: 20),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(title,
                          style: TextStyle(
                              color: accent,
                              fontWeight: FontWeight.w700,
                              fontSize: 14)),
                    ),
                    IconButton(
                      icon: const Icon(Icons.copy_rounded, size: 18),
                      tooltip: 'Copy',
                      onPressed: () {
                        Clipboard.setData(ClipboardData(text: body));
                        ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(content: Text('Copied')));
                      },
                    ),
                  ],
                ),
                Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: SelectableText(body,
                      style: const TextStyle(fontSize: 17, height: 1.5)),
                ),
              ],
            ),
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
              return _message(Icons.cloud_off_rounded, '${streamSnap.error}');
            }
            if (!streamSnap.hasData) return const _Loader();
            return StreamBuilder<QuerySnapshot<Map<String, dynamic>>>(
              stream: streamSnap.data,
              builder: (context, snap) {
                if (snap.hasError) {
                  return _message(Icons.cloud_off_rounded, '${snap.error}');
                }
                if (!snap.hasData) return const _Loader();
                final docs = snap.data!.docs;
                if (docs.isEmpty) {
                  return _message(Icons.history_rounded, 'No scans yet.');
                }
                return ListView.separated(
                  padding: const EdgeInsets.all(20),
                  itemCount: docs.length,
                  separatorBuilder: (_, _) => const SizedBox(height: 12),
                  itemBuilder: (_, i) => _HistoryTile(docs[i].data()),
                );
              },
            );
          },
        ),
      );

  Widget _message(IconData icon, String text) => Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, size: 48, color: kTeal.withValues(alpha: 0.4)),
              const SizedBox(height: 12),
              Text(text, textAlign: TextAlign.center),
            ],
          ),
        ),
      );
}

class _Loader extends StatelessWidget {
  const _Loader();
  @override
  Widget build(BuildContext context) =>
      const Center(child: CircularProgressIndicator(color: kOrange));
}

class _HistoryTile extends StatelessWidget {
  final Map<String, dynamic> data;
  const _HistoryTile(this.data);

  @override
  Widget build(BuildContext context) {
    final ts = (data['createdAt'] as Timestamp?)?.toDate();
    return Card(
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: () => Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => Scaffold(
              appBar: AppBar(title: const Text('Scan')),
              body: ListView(
                padding: const EdgeInsets.all(20),
                children: [
                  ResultView(ScanResult(
                      data['extracted'] ?? '',
                      data['transliterated'] ?? '',
                      data['translated'] ?? '')),
                ],
              ),
            ),
          ),
        ),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(data['extracted'] ?? '',
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                      fontSize: 16, fontWeight: FontWeight.w600)),
              const SizedBox(height: 6),
              Text(data['translated'] ?? '',
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(color: Colors.black.withValues(alpha: 0.6))),
              if (ts != null) ...[
                const SizedBox(height: 8),
                Text(
                    '${ts.day}/${ts.month}/${ts.year}  ${ts.hour.toString().padLeft(2, '0')}:${ts.minute.toString().padLeft(2, '0')}',
                    style: const TextStyle(fontSize: 12, color: kOrange)),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
