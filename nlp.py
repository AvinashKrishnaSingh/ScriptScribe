# nlp.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch




def devanagaritolatin(text, ind_vowels, matras, consonants, diacritics, numerals):
    result = ""
    i = 0
    while i < len(text):
        ch = text[i]

        # Check if the character is a pipe symbol
        if ch == '।':
            # Skip over any consecutive pipe characters ("|" or "||" etc.)
            while i < len(text) and text[i] == '।':
                i += 1
            # Trim any trailing whitespace from the current line and add a newline
            result = result.rstrip() + "\n"
            continue

        # If the character is an independent vowel, output its mapping.
        if ch in ind_vowels:
            result += ind_vowels[ch]

        # If the character is a consonant, handle possible conjuncts or vowel signs.
        elif ch in consonants:
            result += consonants[ch]
            # Look ahead to decide about the inherent vowel.
            if i + 1 < len(text):
                next_char = text[i+1]
                if next_char == '\u094D':  # Virāma: suppress the inherent vowel.
                    i += 1  # Skip the halant so that we don’t add an 'a'
                elif next_char in matras:
                    # Append the mapped matra and skip it.
                    result += matras[next_char]
                    i += 1
                else:
                    # Otherwise, add the inherent vowel "a".
                    result += 'a'
            else:
                result += 'a'

        # In case a vowel sign appears unexpectedly.
        elif ch in matras:
            result += matras[ch]

        # For diacritical marks like anusvāra, visarga, chandrabindu.
        elif ch in diacritics:
            result += diacritics[ch]

        # For numerals.
        elif ch in numerals:
            result += numerals[ch]

        # For punctuation, spaces, and any other characters.
        else:
            result += ch

        i += 1
    return result.strip()

# Independent vowels (for stand-alone vowels)
ind_vowels = {
    'अ': 'a',
    'आ': 'ā',
    'इ': 'i',
    'ई': 'ī',
    'उ': 'u',
    'ऊ': 'ū',
    'ऋ': 'ṛ',
    'ॠ': 'ṝ',
    'ए': 'e',
    'ऐ': 'ai',
    'ओ': 'o',
    'औ': 'au'
}

# Matras (vowel signs used with consonants)
matras = {
    'ा': 'ā',
    'ि': 'i',
    'ी': 'ī',
    'ु': 'u',
    'ू': 'ū',
    'ृ': 'ṛ',
    'े': 'e',
    'ै': 'ai',
    'ो': 'o',
    'ौ': 'au'
}

# Consonants mapping (using diacritics for retroflex letters, etc.)
consonants = {
    'क': 'k',
    'ख': 'kh',
    'ग': 'g',
    'घ': 'gh',
    'ङ': 'ṅ',
    'च': 'c',
    'छ': 'ch',
    'ज': 'j',
    'झ': 'jh',
    'ञ': 'ñ',
    'ट': 'ṭ',
    'ठ': 'ṭh',
    'ड': 'ḍ',
    'ढ': 'ḍh',
    'ण': 'ṇ',
    'त': 't',
    'थ': 'th',
    'द': 'd',
    'ध': 'dh',
    'न': 'n',
    'प': 'p',
    'फ': 'ph',
    'ब': 'b',
    'भ': 'bh',
    'म': 'm',
    'य': 'y',
    'र': 'r',
    'ल': 'l',
    'व': 'v',
    'श': 'ś',
    'ष': 'ṣ',
    'स': 's',
    'ह': 'h',
    # Compound consonants commonly needed:
    'क्ष': 'kṣ',
    'त्र': 'tr',
    'ज्ञ': 'jñ'
}

# Diacritics for special marks (anusvāra, visarga, chandrabindu)
diacritics = {
    'ं': 'ṃ',  # anusvāra
    'ः': 'ḥ',  # visarga
    'ँ': 'ṃ',   # chandrabindu (using ṃ here; some systems use m̐)
    'ॐ': 'Om'
}

# Numerals (Hindi/Devanagari digits)
numerals = {
    '०': '0',
    '१': '1',
    '२': '2',
    '३': '3',
    '४': '4',
    '५': '5',
    '६': '6',
    '७': '7',
    '८': '8',
    '९': '9'
}

def convert_text(text):
    return devanagaritolatin(text, ind_vowels, matras, consonants, diacritics, numerals)



# Load the IndicTrans2 model with custom code
tokenizer = AutoTokenizer.from_pretrained(
    "ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True
)

# Proper translation function
def translate_to_english(text):
    if not text.strip():
        return ""
    # Prepend target language tag for English
    tagged_input = f">>en<< {text.strip()}"
    inputs = tokenizer(tagged_input, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()

# tokenizer = AutoTokenizer.from_pretrained("diabolic6045/Sanskrit-qwen-7B-Translate")
# model     = AutoModelForCausalLM.from_pretrained("diabolic6045/Sanskrit-qwen-7B-Translate")

# def translate_sanskrit(text):
#     inputs  = tokenizer(text, return_tensors="pt")
#     output  = model.generate(**inputs, max_new_tokens=200)
#     return tokenizer.decode(output[0], skip_special_tokens=True)