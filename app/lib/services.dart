import 'dart:io';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import 'package:google_mlkit_translation/google_mlkit_translation.dart';

import 'transliterator.dart';

class ScanResult {
  final String extracted;
  final String transliterated;
  final String translated;
  const ScanResult(this.extracted, this.transliterated, this.translated);
}

/// On-device OCR (Devanagari) -> IAST transliteration -> English translation.
class ScriptScribeService {
  final _recognizer = TextRecognizer(script: TextRecognitionScript.devanagiri);
  final _translator = OnDeviceTranslator(
    sourceLanguage: TranslateLanguage.hindi,
    targetLanguage: TranslateLanguage.english,
  );
  final _modelManager = OnDeviceTranslatorModelManager();

  Future<ScanResult> process(File image,
      {void Function(String stage)? onStage}) async {
    onStage?.call('Reading the text…');
    final recognized =
        await _recognizer.processImage(InputImage.fromFile(image));
    final text = recognized.text.trim();
    if (text.isEmpty) return const ScanResult('', '', '');

    onStage?.call('Transliterating…');
    final latin = devanagariToLatin(text);

    // Sanskrit is not supported by ML Kit; Hindi is the closest model.
    var english = '';
    try {
      for (final code in [TranslateLanguage.hindi, TranslateLanguage.english]) {
        if (!await _modelManager.isModelDownloaded(code.bcpCode)) {
          onStage?.call('Downloading translation model (first time only)…');
          await _modelManager.downloadModel(code.bcpCode);
        }
      }
      onStage?.call('Translating to English…');
      english = await _translator.translateText(text);
    } catch (e) {
      english = 'Translation unavailable: $e';
    }
    return ScanResult(text, latin, english);
  }

  void dispose() {
    _recognizer.close();
    _translator.close();
  }
}

class HistoryService {
  final _db = FirebaseFirestore.instance;
  final _auth = FirebaseAuth.instance;

  Future<String> _uid() async {
    final user = _auth.currentUser ?? (await _auth.signInAnonymously()).user;
    return user!.uid;
  }

  Future<CollectionReference<Map<String, dynamic>>> _col() async =>
      _db.collection('users').doc(await _uid()).collection('scans');

  Future<void> save(ScanResult r) async {
    final col = await _col();
    await col.add({
      'extracted': r.extracted,
      'transliterated': r.transliterated,
      'translated': r.translated,
      'createdAt': FieldValue.serverTimestamp(),
    });
  }

  Future<Stream<QuerySnapshot<Map<String, dynamic>>>> stream() async =>
      (await _col()).orderBy('createdAt', descending: true).snapshots();
}
