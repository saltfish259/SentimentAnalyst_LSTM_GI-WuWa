# Sentiment Analyst Using LSTM - Razif Zulvikar Hatuwe

🔗 **Link Dashboard**: [Klik di sini](https://sentimentanalystlstmgi-wuwa-wwvypmgve3sitf8mryizy3.streamlit.app/)

## A. Domain Proyek

### Latar Belakang

Genshin Impact dan Wuthering Waves merupakan dua game action RPG populer yang memiliki basis pemain global dengan jumlah unduhan yang sangat tinggi di Google Play Store. Kedua game ini kerap menjadi bahan diskusi hangat di kalangan gamer, baik dalam bentuk ulasan positif seperti grafis dan gameplay yang menarik, maupun kritik terhadap bug, performa game, atau sistem gacha.

Dengan banyaknya komentar dari pengguna, pengembang dan tim pemasaran perlu mengetahui sentimen dominan dari para pemain terhadap game mereka. Namun, membaca ribuan komentar secara manual tentu tidak efisien. Oleh karena itu, dibutuhkan pendekatan berbasis **Sentiment Analysis** untuk mengolah opini-opini tersebut secara otomatis dan sistematis.

### Mengapa dan Bagaimana Masalah Ini Diselesaikan

Masalah ini diselesaikan dengan pendekatan berbasis **Natural Language Processing (NLP)**, khususnya dengan menggunakan model **Long Short-Term Memory (LSTM)** yang mampu memahami konteks dalam teks. Untuk membantu proses preprocessing, digunakan library **NLTK**. Dengan model ini, kita dapat mengklasifikasikan komentar pengguna menjadi sentimen **positif** atau **negatif**, sehingga memberikan gambaran umum persepsi masyarakat terhadap masing-masing game.

---

## B. Business Understanding

### Problem Statements

Bagaimana cara mengetahui persepsi pengguna terhadap game *Genshin Impact* dan *Wuthering Waves* berdasarkan komentar di Google Play Store, dan mengklasifikasikannya menjadi sentimen positif atau negatif secara otomatis?

### Goals

Mengetahui distribusi sentimen (positif dan negatif) dari komentar pengguna terhadap kedua game, serta mengevaluasi performa model klasifikasi berbasis LSTM yang digunakan.

### Solution

Membangun model deep learning berbasis LSTM menggunakan data komentar dari Google Play Store dan teknik NLP, untuk melakukan analisis sentimen terhadap dua game tersebut.

---

## C. Data Understanding

### Sumber Data

Data diperoleh menggunakan library **Google Play Scraper**. Jumlah data yang diambil adalah:

* **5000 komentar**

```python
from google_play_scraper import reviews

app_id = "Game.Game"
num_reviews = 5000
result, continuation_token = reviews(
    app_id,
    lang='en',
    country='us',
    count=num_reviews,
)
```

---

## D. Data Preparation

Langkah-langkah preprocessing dilakukan untuk membersihkan data dari noise dan mempersiapkannya agar dapat digunakan oleh model LSTM. Berikut adalah tahapan pembersihan teks:

```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['Clean_Reviews'] = df['Review'].apply(clean_text)
```

Setelah data dibersihkan, langkah selanjutnya adalah membagi data menjadi data latih dan data uji:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## E. Modeling

Model LSTM dibangun menggunakan Keras Sequential API. Berikut adalah arsitektur model yang digunakan:

```python
model = Sequential([
    Embedding(10000, 64, input_length=100),
    LSTM(64),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

Model dilatih menggunakan fungsi loss `binary_crossentropy` dan optimizer `adam`.

---

## F. Evaluasi

Model diuji pada data uji dan menghasilkan hasil evaluasi sebagai berikut:

```
Akurasi: 88.35%
Loss: 0.3558

Classification Report:
              precision    recall  f1-score   support

           0       0.71      0.64      0.67       158
           1       0.92      0.94      0.93       692

    accuracy                           0.88       850
   macro avg       0.81      0.79      0.80       850
weighted avg       0.88      0.88      0.88       850
```

Model memiliki performa yang tinggi dalam mengklasifikasikan sentimen positif, dan cukup baik dalam mengidentifikasi sentimen negatif.

---

## G. Kesimpulan

Secara keseluruhan, proyek ini berhasil menjawab problem statement yang telah dirumuskan. Dengan menggunakan pendekatan LSTM dan teknik NLP, model yang dibangun mampu mengklasifikasikan komentar pengguna dengan akurasi tinggi. Model ini efektif dalam mengungkap persepsi pengguna terhadap dua game populer, yaitu Genshin Impact dan Wuthering Waves. Hasil ini menunjukkan bahwa metode yang digunakan cukup andal dalam menganalisis opini berbasis teks dalam skala besar dan dapat dijadikan dasar untuk pengambilan keputusan lebih lanjut oleh pengembang game.

---

## H. Dashboard Interaktif

Sebagai pelengkap dari proyek ini, telah dibangun sebuah dashboard interaktif menggunakan **Streamlit** untuk memvisualisasikan hasil analisis sentimen secara langsung.

🔗 **Link Dashboard**: [Klik di sini](https://sentimentanalystlstmgi-wuwa-wwvypmgve3sitf8mryizy3.streamlit.app/)

### Alur Dashboard:

1. **Pemilihan Game**

   * Pengguna dapat memilih salah satu game (Genshin Impact atau Wuthering Waves) melalui sidebar.

2. **Visualisasi Sentimen (Section 1)**

   * Menampilkan **pie chart** dan **bar chart** yang menggambarkan proporsi sentimen positif dan negatif dari komentar pengguna terhadap game yang dipilih.

3. **Word Cloud (Section 2)**

   * Terdapat dua word cloud yang memperlihatkan kata-kata yang sering muncul dalam komentar positif dan komentar negatif.

4. **Uji Coba Model (Section 3)**

   * Pengguna dapat mengetikkan kalimat secara manual, dan sistem akan memprediksi sentimen dari input tersebut sebagai **positif** atau **negatif**.

Dashboard ini memberikan pengalaman eksploratif sekaligus pembuktian terhadap hasil model yang telah dibangun. Ini sangat berguna bagi pemangku kepentingan untuk memahami tren opini dan melakukan analisis lebih lanjut.
