import os
import pandas as pd
import re
import string
import torch
from flask import Flask, render_template, request, send_file
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Konfigurasi folder upload
app.config['UPLOAD_FOLDER'] = 'static/uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

#  1. CONFIG & LOAD MODEL
MODEL_PATH = "Model IndoBERT" 

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    print("Model IndoBERT Berhasil Dimuat.")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    MODEL_NAME = "indobenchmark/indobert-base-p1"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

#  2. KAMUS ALAY & SLANG
kamus_alay = {
    "ga": "tidak", "gak": "tidak", "ngga": "tidak", "nggak": "tidak", "tdk": "tidak",
    "gk": "tidak", "g": "tidak", "tp": "tapi", "tpi": "tapi", "klo": "kalau",
    "kalo": "kalau", "yg": "yang", "jd": "jadi", "jdi": "jadi", "krn": "karena",
    "karna": "karena", "dgn": "dengan", "dg": "dengan", "bkn": "bukan",
    "bgt": "banget", "bngt": "banget", "bgtt": "banget", "bagus": "bagus",
    "bgus": "bagus", "jlk": "jelek", "jele": "jelek", "ancur": "hancur",
    "kren": "keren", "bnr": "benar", "bener": "benar", "sm": "sama",
    "bnyk": "banyak", "sdh": "sudah", "udah": "sudah", "udh": "sudah",
    "blm": "belum", "belom": "belum", "gpp": "tidak apa apa",
    "sy": "saya", "gw": "saya", "gue": "saya", "gua": "saya", "aku": "saya",
    "lu": "kamu", "lo": "kamu", "loe": "kamu", "km": "kamu", "u": "kamu",
    "dr": "dari", "dri": "dari", "pd": "pada", "dlm": "dalam", "ngab": "bang",
    "cegil": "perempuan obsesif", "cogil": "laki laki obsesif", "fomo": "ikut ikutan",
    "redflag": "berbahaya", "red flag": "berbahaya", "greenflag": "baik",
    "ytjt": "yang tahu tahu saja", "fyp": "rekomendasi", "anjay": "keren",
    "anjir": "parah", "pls": "tolong", "jujurly": "sejujurnya",
    "sengaruh": "berpengaruh", "caper": "cari perhatian", "baper": "bawa perasaan",
    "gaje": "tidak jelas", "mager": "malas", "pargoy": "joget",
    "spill": "bocorkan", "salty": "kesal", "toxic": "beracun",
    "cringe": "aneh", "vibes": "suasana", "worth it": "pantas"
}

#  3. FUNGSI PREPROCESSING

def clean_text(text):
    if not isinstance(text, str) or text is None:
        return ""
    # Hapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Hapus mention & hashtag
    text = re.sub(r'@\w+|#\w+', '', text)
    # Hapus emoji & karakter non-ascii
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    # Hapus tanda baca
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))
    # Hapus karakter selain huruf a-z
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Normalisasi huruf berulang (contoh: "loooove" jadi "love")
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    # Lowercase & hapus spasi berlebih
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def normalisasi_alay(text):
    words = text.split()
    normalized_words = [kamus_alay.get(w, w) for w in words]
    return " ".join(normalized_words)

#  4. PREDICTION FUNCTION 
def predict_sentiment(text):
    # Tahap 1: Cleaning
    cleaned = clean_text(text)
    # Tahap 2: Normalisasi
    final_text = normalisasi_alay(cleaned)
    
    # Tokenisasi & Prediksi
    inputs = tokenizer(final_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)
    
    # Mapping Label: 0 = non-bullying, 1 = bullying
    sentiment_labels = ["non-bullying", "bullying"]
    return sentiment_labels[predicted_class.item()], confidence.item()

#  5. ROUTES 
@app.route('/', methods=['GET', 'POST'])
def index():
    active_tab = 'manual' 
    manual_result = None
    results = None
    summary = ""
    pie_chart_data = [0, 0] 
    download_link = None

    if request.method == 'POST':
        if 'manual_text' in request.form:
            active_tab = 'manual'
            raw_text = request.form['manual_text']
            label, score = predict_sentiment(raw_text)
            
            manual_result = {
                'text': raw_text,
                'sentiment': label,
                'confidence': round(score * 100, 2)
            }

        elif 'file' in request.files:
            active_tab = 'batch'
            file = request.files['file']
            if file and file.filename.endswith('.csv'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                
                df = pd.read_csv(filepath)
                col_name = 'text_komentar' if 'text_komentar' in df.columns else df.columns[0]

                # 1. Jalankan fungsi preprocessing dan simpan ke kolom 'cleaned_text'
                df['cleaned_text'] = df[col_name].apply(clean_text).apply(normalisasi_alay)
                
                # 2. Prediksi menggunakan teks yang sudah bersih
                predictions = df['cleaned_text'].apply(predict_sentiment)
                
                # 3. Masukkan hasil prediksi ke dataframe
                df['predicted_sentiment'] = [p[0] for p in predictions]
                df['confidence'] = [round(p[1] * 100, 2) for p in predictions]
                
                counts = df['predicted_sentiment'].value_counts()
                bull_count = int(counts.get('bullying', 0))
                non_bull_count = int(counts.get('non-bullying', 0))
                
                results = df.to_dict(orient='records')
                pie_chart_data = [bull_count, non_bull_count]
                summary = f"Total Data: {len(df)} | Bullying: {bull_count} | Non-Bullying: {non_bull_count}"
                
                output_filename = f"hasil_analisis_{file.filename}"
                df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], output_filename), index=False)
                download_link = f"/download/{output_filename}"

    return render_template('index.html', 
                           active_tab=active_tab,
                           manual_result=manual_result,
                           results=results,
                           summary=summary,
                           pie_chart_data=pie_chart_data,
                           download_link=download_link)

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5001)