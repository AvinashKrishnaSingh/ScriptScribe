/// Devanagari -> IAST transliteration (port of nlp.py).
const _vowels = {
  'अ': 'a', 'आ': 'ā', 'इ': 'i', 'ई': 'ī', 'उ': 'u', 'ऊ': 'ū',
  'ऋ': 'ṛ', 'ॠ': 'ṝ', 'ऌ': 'ḷ', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
};

const _matras = {
  'ा': 'ā', 'ि': 'i', 'ी': 'ī', 'ु': 'u', 'ू': 'ū', 'ृ': 'ṛ', 'ॄ': 'ṝ',
  'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
};

const _consonants = {
  'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ṅ',
  'च': 'c', 'छ': 'ch', 'ज': 'j', 'झ': 'jh', 'ञ': 'ñ',
  'ट': 'ṭ', 'ठ': 'ṭh', 'ड': 'ḍ', 'ढ': 'ḍh', 'ण': 'ṇ',
  'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
  'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
  'य': 'y', 'र': 'r', 'ल': 'l', 'ळ': 'ḷ', 'व': 'v',
  'श': 'ś', 'ष': 'ṣ', 'स': 's', 'ह': 'h',
  'ड़': 'ṛ', 'ढ़': 'ṛh', 'क़': 'q', 'ख़': 'kh', 'ग़': 'ġ', 'ज़': 'z', 'फ़': 'f',
};

const _diacritics = {'ं': 'ṃ', 'ः': 'ḥ', 'ँ': 'ṃ', 'ॐ': 'om', 'ऽ': "'"};

const _numerals = {
  '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
  '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
};

const _virama = '\u094D';
const _nukta = '\u093C';

String devanagariToLatin(String text) {
  final chars = text.runes.map(String.fromCharCode).toList();
  final out = StringBuffer();
  var i = 0;
  while (i < chars.length) {
    final ch = chars[i];

    // danda / double danda end a line
    if (ch == '।' || ch == '॥') {
      while (i < chars.length && (chars[i] == '।' || chars[i] == '॥')) {
        i++;
      }
      final s = out.toString().trimRight();
      out
        ..clear()
        ..write(s)
        ..write('\n');
      continue;
    }

    if (_vowels.containsKey(ch)) {
      out.write(_vowels[ch]);
    } else if (_consonants.containsKey(ch)) {
      var base = ch;
      // fold a following nukta into the consonant when a precomposed form exists
      if (i + 1 < chars.length && chars[i + 1] == _nukta) {
        final combined = '$ch$_nukta';
        final composed = combined.replaceAll('ड$_nukta', 'ड़').replaceAll('ढ$_nukta', 'ढ़');
        if (_consonants.containsKey(composed)) base = composed;
        i++;
      }
      out.write(_consonants[base]);
      final next = i + 1 < chars.length ? chars[i + 1] : null;
      if (next == _virama) {
        i++;
      } else if (next != null && _matras.containsKey(next)) {
        out.write(_matras[next]);
        i++;
      } else {
        out.write('a');
      }
    } else if (_matras.containsKey(ch)) {
      out.write(_matras[ch]);
    } else if (_diacritics.containsKey(ch)) {
      out.write(_diacritics[ch]);
    } else if (_numerals.containsKey(ch)) {
      out.write(_numerals[ch]);
    } else if (ch == _nukta || ch == _virama) {
      // stray combining marks are dropped
    } else {
      out.write(ch);
    }
    i++;
  }
  return out.toString().trim();
}
