# server.py
import os
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from waitress import serve
from ocr import refine_image, extract_text
from nlp import convert_text, translate_to_english


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['REFINED_FOLDER'] = 'refined'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REFINED_FOLDER'], exist_ok=True)

@app.route('/')
def loading():
    return render_template('loading.html')

@app.route('/index')
def index():
    return render_template('index.html')


# Serve uploaded and refined images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/refined/<filename>')
def refined_file(filename):
    return send_from_directory(app.config['REFINED_FOLDER'], filename)

# Single /upload route that handles saving and refining the image
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    # Save the file to the uploads directory
    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Refine the image
    refined_filename = 'refined_' + filename
    refined_image_path = os.path.join(app.config['REFINED_FOLDER'], refined_filename)
    if not refine_image(file_path, refined_image_path):
        return jsonify({"error": "Image refinement failed"}), 500

    # Extract text using OCR
    extracted_text = extract_text(file_path)

     # Translate the extracted text
    devtolat = convert_text(extracted_text)

    translated = translate_to_english(extracted_text)


    image_url = url_for('refined_file', filename=refined_filename)

    # Render the results page and pass the extracted text
    return render_template('results.html', extracted_text=extracted_text, devtolat=devtolat, translated=translated,image_url=image_url)



if __name__ == '__main__':
    serve(app, host='0.0.0.0', port='8080')
