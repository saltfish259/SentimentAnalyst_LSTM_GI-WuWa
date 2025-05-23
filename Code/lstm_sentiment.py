# -*- coding: utf-8 -*-
"""LSTM_Sentiment.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BROJWiEmAgRV_tluMXNXdzvavdKeH_ow
"""

import re
import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf

nltk.download('vader_lexicon')

"""Berfungsi untuk mengunduh leksikon VADER (**Valence Aware Dictionary and Sentiment Reasoner**) dari pustaka NLTK (**Natural Language Toolkit**). Digunakan dalam analisis sentiment texks."""

df = pd.read_csv("Wuthering_Reviews.csv")

"""Berfungsi untuk membaca file CSV bernama `Wuthering_Reviews.csv` dan membuat isi kedalam DataFrame bernama df menggunakan pustaka pandas (`pd`)."""

df['Review'] = df['Review'].astype(str)

"""Berfungsi untuk mengonversi seluruh nilai dalam kolom `Review` menjadi tipe data string (`str`), lalu menyimpan kembali ke dalam kolom yang sama dalam DataFrame `df`"""

sia = SentimentIntensityAnalyzer()

"""Berfungsi untuk membuat sebuah objek `SentimentIntensityAnalyzer` dari pustaka `nltk.sentiment.vader`. Objek ini digunakan untuk menganalisis sentimen dari teks, seperti kalimat atau ulasan."""

def get_sentiment(text):
    score = sia.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 1
    elif score['compound'] <= -0.05:
        return 0
    else:
        return None

"""Bertujuan untuk mengklasifikasikan sentimen dari suatu teks sebagai **positif(1)** dan **Negatif(0)** berdasarkan skor compound dari VADER.

Penjelasan:
1. `score = sia.polarity_scores(text)`
Mengambil skor sentiment dari teks menggunakan VADER. Skor ini berbentuk dictionary seperti ini:


```
{'neg': 0.1, 'neu': 0.7, 'pos': 0.2, 'compound': 0.3612}
```

2. `if score['compound'] >= 0.05:`
Jika skor gabungan (compound) cukup tinggi -> dianggap **POSITIF** -> return `1`

3. `elif score['compound'] <= -0.05:`
Jika skor compound cukup rendah -> dianggap **NEGATIF** -> return `0`.

4. else:
Jika skor berada di antara -0.05 dan 0.05 -> dianggap **NETRAL** -> return `None`



"""

df['Sentiment'] = df['Review'].apply(get_sentiment)
df = df.dropna(subset=['Sentiment'])

"""Digunakan untuk mengklasifikasikan sentimen setiap ulasan dan menghapus baris yang tidak memilki sentimen yang jelas (netral) dari DataFrame

Penjelasan:
1.  `df['Sentiment'] = df['Review'].apply(get_sentiment)`
- Menerapkan fungsi `get_sentiment` (yang telah dibuat sebelumnya) ke setiap elemen dalam kolom `Review`.
- Hasil klasifikasi (0 = negatif, 1 = positif, None = netral) disimpan di kolom baru bernama `Sentiment`
"""

def clean_text(text):
  text = str(text).lower()
  text = re.sub(r"http\S+|www\S+|@\w+|#", "", text)
  text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
  text = re.sub(r"\s+", " ", text).strip()
  return text

df['Clean_Reviews'] = df ['Review'].apply(clean_text)

"""Digunakan untuk membersihkan teks ulasan sebelum diproses lebih lanjut dalam pipeline NLP.

Penjelasan:
1. `text = str(text).lower()`
Mengonversi teks menjadi string (untuk jaga jaga jika ada nilai non string) dan semua huruf ke huruf kecil, agar standar.

2. `text = re.sub(r"http\S+|www\S+|@\w+|#", "", text)`
Menghapus tautan (URL), mention (@username), dan simbol tagar (#).

3. `text = re.sub(r"[^a-zA-Z0-9\s]", "", text)`
Menghapus tanda baca dan karakter non-alfanumerik, menyisakan hanya huruf, angka, dan spasi.

4. `text = re.sub(r"\s+", " ", text).strip()`
Menghapus spasi berlebih (termasuk tab dan newline), dan menghilangkan spasi diawal/akhir teks
"""

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['Clean_Reviews'])

"""Digunakan untuk mempersiapkan data teks agar bisa diubah menjadi urutan angka yang dapat diproses oleh model machine learning.

Penjelasan:
1. `tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')`
`Tokenizer` adalah kelas dari `tensorflow.keras.preprocessing.text` yang digunakan untuk:
- Membuat indeks kata berdasarkan frekuensi kemunculan.
- mengubah teks menjadi urutan angka (integer sequences).

2. `num_words=10000`:
- Hanya **10.000 kata teratas** yang paling sering muncul yang akan dipertahankan dalam kamustokenizer.
- kata diluar daftar ini akan dianggap sebagai kata yang jarang (rare words).

3. `oov_token='<OOV>'`:
- Singkatan dari *Out Of Vocabulary*.
- Kata - kata yang tidak ditemukan selama `fit` akan digantikan dengan token khusus saat digunakan nanti `(text_to_sequences())`.
- Membantu menjaga model tetap stabils aat menerima kata baru di data uji / prediksi.
"""

X = tokenizer.texts_to_sequences(df['Clean_Reviews'])
X = pad_sequences(X, maxlen=100)

"""Digunakan untuk mengonversi teks menjadi urutan angka dan memastikan semua urutan memiliki panjang yang sama agr bisa diproses oleh model.

Penjelasan:
1. `X = tokenizer.texts_to_sequences(df['Clean_Reviews'])`
- Mengubah setiap review bersih (`Clean_Reviews`) menjadi daftar angka.
- Angka tersebut adalah indeks kata berdasarkan kamus/tokenizer yang sudah dibuat sebelumnya.
- Kata-kata yang tidak dikenali (tidak masuk 10.000 kata teratas) akan diganti dengan indeks dari `<OOV>` token.
contoh:


```
["this book is great"] → [[12, 345, 6, 789]]
```

2. X = pad_sequences(X, maxlen=100)
- Mengubah semua urutan angka menjadi panjang tetap (dalam hal ini, 100 token).
- jika suatu runtutan lebih pendek dari 100, maka akan **ditambahkan padding (nol)** didepan (`default: padding='pre`).
- jika lebih panjang dari 100, maka urutan akan dipotong dari awal(`default: trauncating='pre'`)
"""

y = df['Sentiment'].astype(int)

"""Berfungsi untuk mengonversi kolom Sentiment menjadi tipe data integer (1 untuk positif dan 0 untuk negatif) yangdigunakan sebagai label target dalam model pembelajaran mesin.

Penjelasan:
1. `df['Sentiment']`: Kolom `Sentiment` yang sebelumnya berisi nilai klasifikasi sentimen (`1`, `0`, atau `None`).
2. `.astype(int)`: Mengonversi nilai dalam kolom `Sentiment` menjadi tipe integer.

- Positif (`1`).

- Negatif (`0`).
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Digunakan untuk membagi data menjadi set pelatihan dan set pengujian.

Penjelasan:
1. `train_test_split` adalah fungsi dari pustaka `sklearn.model_selection` yang digunakan untuk membagi data menjadi dua bagian:
- Set pelatihan (`X_train`, `y_train`): Digunakan untuk melatih model.
- Set penguji (`X_test`, `y_test`): Digunakan untuk mengevaluasi kinerja model setelah pelatihan.
"""

model = Sequential([
    Embedding(10000, 64, input_length=100),
    LSTM(64),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

"""Digunakan untuk membangun model jaringan saraf (neural network) dengan menggunakan keras (bagian dari Tensorflow) untuk klasifikasi sentiment.

Penjelasan:
1. `Embedding(10000, 64, input_length=100)`
- Tujuan: mengubah urutan angka (token yang dihasilkan sebelumnya) menjadi vektor kata berdimensi lebih rendah.
- `10000`: ukuran maksimum kata yang akan dipertimbangkan oleh model (berarti hanya 10.000 kata teratas yang akan diproses).
- `64`: Dimensi ruang vektor untuk setiap kata (setiap kata akandipetakan menjadi vektor dengan panjang 64).
- `input_length=100`: panjang input untuks etiap contoh (dalam hal ini, 100 token). Artinya setiap input harus memiliki panjang 100 (padded sequences).
2. `LSTM(64)`
- Tujuan: menggunakan **Long Short-Term Memory (LSTM)**, jenis **Recurrent Neural Network (RNN)**, untuk menangkap informasi kontekstual dalam urutan data(seperti hubungan antara kata-kata dalam kalimat).
- `64`: Jumlah untuk dalam layer LSTM, yang menentukan jumlah neuron yang akan digunakan dalam lapisan LSTM.
3. `Dropout(0.5)`
- Tujuan: Dropout digunakan untuk mengurangi overfitting dengan "menonaktifkan" sebagaian neuron selama pelatihan secara acak.
- `0.5`: artinya 50% neuron akan dimatikan selama pelatihan untuk mencegah model terlalu mengandalkan fitur-fitur tertentu yang mungkin hanya berlaku untuk data pelatihan saja.
4. `Dense(1, activation='sigmoid')`
- Tujuan: Lapisan terakhir dari model.
- `1`: model hanya menghasilkan satu output (karena kita hanya memiliki dua kelas: positif dan negatif).
- `activation='sigmoid'`: Fungsi aktivitas sigmoid menghasilkan output antara 0 dan 1, yang cocok untuk masalah klasifikasi biner, jika outout lebih besar dari 0.5, itu berarti sentimen positif (1), jika kurang dari 0.5, itu berarti sentimen negatif (0).
"""

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

"""Berfungsi untuk mengkompilasi model dan menampilkan ringkasan arsitektur model.

Penjelasan:
1. `model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])`
- `loss='binary_crossentropy'`: Fungsi kerugian yang digunakan untuk klasifikasi biner (positif/negatif).
- `optimizer='adam'`: Optimizer yang digunakan untuk mengupdate bobot model selama pelatihan.
- `metrics=['accuracy']`: Menghitung akurasi sebagai metrik evaluasi selama pelatihan dan pengujian.

2. `model.summary()`
- Tujuan: Menampilkan ringkasan arsitektur model.

"""

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

"""Digunakan untuk melatih model menggunakan data pelatihan (`X_train` dan `y_train`) dan mengevaluasi kinerjanya dengan data pengujian (`X_test` dan `y_test`) selama pelatihan.

Penjelasan:
1. `X_train` dan `y_train`:
- `X_train`: Data fitur untuk pelatihan (urutan angka dari teks yang sudah diproses).
- `y_train`: Label target untuk pelatihan (sentimen positif atau negatif).

2. `epochs=5`:
- Menentukan jumlah iterasi untuk pelatihan model di seluruh data.

3. `batch_size=32`:
- Batch size adalah jumlah sampel yang diproses sebelum model melakukan pembaruan bobot.


"""

y_pred = (model.predict(X_test) >= 0.5).astype(int)
print("Classification Report:")
print(classification_report(y_test, y_pred))

"""Digunakan untuk melakukan prediksi pada data pengujian dan kemudian menampilkan laporan klasifikasi untuk mengevaluasi kinerja model."""

df["Predicted_Sentiment"] = (model.predict(X) >= 0.5).astype(int)
df[["Review", "Clean_Reviews", "Predicted_Sentiment"]].to_csv("Sentiment_Wuthering.csv", index=False)
print("✅ Sentiment_Game.csv berhasil disimpan.")

model.save("sentiment_model.keras")
print("✅ Model berhasil disimpan.")

import json
tokenizer_json = tokenizer.to_json()
with open("tokenizer_lstm.json", "w") as json_file:
    json.dump(tokenizer_json, json_file)