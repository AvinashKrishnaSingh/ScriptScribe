import 'package:flutter_test/flutter_test.dart';
import 'package:scriptscribe/transliterator.dart';

void main() {
  test('transliterates Devanagari to IAST', () {
    expect(devanagariToLatin('धर्मक्षेत्रे'), 'dharmakṣetre');
    expect(devanagariToLatin('राम'), 'rāma');
    expect(devanagariToLatin('ॐ'), 'om');
    expect(devanagariToLatin('सत्यम् १२'), 'satyam 12');
  });
}
