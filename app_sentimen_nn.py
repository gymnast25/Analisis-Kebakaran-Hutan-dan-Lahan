"""Dashboard Analisis Sentimen Neural Network — Topik: Kebakaran Hutan.
Jalankan:  streamlit run app_sentimen_nn.py
"""
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

st.set_page_config(page_title="Sentimen Karhutla - Neural Network", page_icon="🔥", layout="wide")

URUTAN = ["positif", "netral", "negatif"]
WARNA = {"positif": "#2ca02c", "netral": "#7f7f7f", "negatif": "#d62728"}
CMAP = {"positif": "Greens", "netral": "Greys", "negatif": "Reds"}
EMOJI = {"positif": "😊", "netral": "😐", "negatif": "😡"}

KATA_POSITIF = ["aman", "lega", "selamat", "bantuan", "pulih", "padam", "berhasil",
                "terkendali", "sigap", "membaik", "tertangani", "terbantu", "apresiasi"]
KATA_NEGATIF = ["asap", "parah", "rusak", "bahaya", "kabut", "hangus", "ispa", "korban",
                "meluas", "sesak", "polusi", "ludes", "mengungsi", "berbahaya", "kerugian"]


def bersihkan(teks):
    teks = str(teks)
    teks = re.sub(r"http\S+|www\.\S+", " ", teks)
    teks = re.sub(r"@\w+", " ", teks)
    teks = re.sub(r"#\w+", " ", teks)
    teks = re.sub(r"\d+", " ", teks)
    teks = re.sub(r"[^a-zA-Z\s]", " ", teks)
    teks = teks.lower()
    return re.sub(r"\s+", " ", teks).strip()


def beri_label(teks):
    token = teks.split()
    pos = sum(t in KATA_POSITIF for t in token)
    neg = sum(t in KATA_NEGATIF for t in token)
    return "positif" if pos > neg else "negatif" if neg > pos else "netral"


@st.cache_data
def siapkan_data(sumber):
    df = pd.read_csv(sumber)
    df["clean_text"] = df["full_text"].apply(bersihkan)
    df["sentimen"] = df["clean_text"].apply(beri_label)
    return df


@st.cache_resource
def latih_model(teks, label, hidden, max_iter):
    X_tr, X_te, y_tr, y_te = train_test_split(
        pd.Series(teks), pd.Series(label), test_size=0.2, random_state=42, stratify=label)
    tfidf = TfidfVectorizer()
    Xtr = tfidf.fit_transform(X_tr)
    mlp = MLPClassifier(hidden_layer_sizes=(hidden,), activation="relu", solver="adam",
                        max_iter=max_iter, random_state=42)
    mlp.fit(Xtr, y_tr)
    y_pred = mlp.predict(tfidf.transform(X_te))
    return tfidf, mlp, list(y_te), list(y_pred)


# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Pengaturan")
unggah = st.sidebar.file_uploader("Unggah CSV lain (opsional)", type="csv")
hidden = st.sidebar.slider("Neuron tersembunyi", 10, 300, 100, 10)
max_iter = st.sidebar.slider("Maksimum iterasi", 100, 600, 300, 50)
st.sidebar.caption("Kolom wajib: created_at, username, full_text")

try:
    df = siapkan_data(unggah if unggah is not None else "tweet_data.csv")
except Exception as e:
    st.error(f"Gagal membaca data: {e}")
    st.stop()

if not {"created_at", "username", "full_text"}.issubset(df.columns):
    st.error("CSV harus memiliki kolom created_at, username, full_text.")
    st.stop()

tfidf, mlp, y_test, y_pred = latih_model(
    tuple(df["clean_text"]), tuple(df["sentimen"]), hidden, max_iter)

# ---------------- Header ----------------
st.title("🔥 Analisis Sentimen Kebakaran Hutan")
st.caption("Pipeline: cleansing → pelabelan rule-based → TF-IDF → Neural Network (MLP)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total tweet", len(df))
for col, k in zip((c2, c3, c4), URUTAN):
    col.metric(f"{EMOJI[k]} {k.capitalize()}", int((df["sentimen"] == k).sum()))

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📄 Data", "📊 Distribusi", "☁️ WordCloud", "🧠 Evaluasi Model", "✍️ Prediksi Teks"])

# ---------------- Tab 1: data ----------------
with tab1:
    st.subheader("Data setelah cleansing & pelabelan")
    filter_s = st.multiselect("Filter sentimen", URUTAN, default=URUTAN)
    st.dataframe(df[df["sentimen"].isin(filter_s)][
        ["created_at", "username", "full_text", "clean_text", "sentimen"]],
        width="stretch", height=420)

# ---------------- Tab 2: distribusi ----------------
with tab2:
    jumlah = df["sentimen"].value_counts().reindex(URUTAN).fillna(0).astype(int)
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(jumlah.index, jumlah.values, color=[WARNA[k] for k in jumlah.index])
        for i, v in enumerate(jumlah.values):
            ax.text(i, v + 1, str(v), ha="center")
        ax.set_title("Bar Chart"); ax.set_ylabel("Jumlah tweet")
        st.pyplot(fig)
    with b:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(jumlah.values, labels=jumlah.index, autopct="%1.1f%%", startangle=90,
               colors=[WARNA[k] for k in jumlah.index])
        ax.set_title("Pie Chart")
        st.pyplot(fig)

# ---------------- Tab 3: wordcloud ----------------
with tab3:
    cols = st.columns(3)
    for col, k in zip(cols, URUTAN):
        with col:
            teks = " ".join(df.loc[df["sentimen"] == k, "clean_text"])
            st.markdown(f"**{EMOJI[k]} {k.capitalize()}**")
            if teks.strip():
                wc = WordCloud(width=500, height=350, background_color="white",
                               colormap=CMAP[k], max_words=60).generate(teks)
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("Tidak ada data.")

# ---------------- Tab 4: evaluasi ----------------
with tab4:
    metrik = pd.DataFrame({
        "Metrik": ["Akurasi", "Presisi", "Recall", "F1-Score"],
        "Nilai": [accuracy_score(y_test, y_pred),
                  precision_score(y_test, y_pred, average="weighted", zero_division=0),
                  recall_score(y_test, y_pred, average="weighted", zero_division=0),
                  f1_score(y_test, y_pred, average="weighted", zero_division=0)]})
    a, b = st.columns(2)
    with a:
        st.subheader("Tabel metrik (weighted)")
        st.dataframe(metrik.style.format({"Nilai": "{:.4f}"}), width="stretch")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(metrik["Metrik"], metrik["Nilai"], color="#1f77b4")
        ax.set_ylim(0, 1.05)
        for i, v in enumerate(metrik["Nilai"]):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
        st.pyplot(fig)
    with b:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=URUTAN)
        fig, ax = plt.subplots(figsize=(5, 4.2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=URUTAN,
                    yticklabels=URUTAN, ax=ax)
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
        st.pyplot(fig)
    st.caption("Label berasal dari kamus kata dan fitur model adalah kata yang sama, "
               "sehingga skor tinggi wajar terjadi dan tidak otomatis berarti model memahami sentimen.")

# ---------------- Tab 5: prediksi ----------------
with tab5:
    st.subheader("Prediksi sentimen teks baru")
    teks_baru = st.text_area("Masukkan teks", "Asap karhutla makin parah, warga sesak napas", height=100)
    if st.button("Prediksi", type="primary"):
        bersih = bersihkan(teks_baru)
        if not bersih:
            st.warning("Teks kosong setelah dibersihkan.")
        else:
            vek = tfidf.transform([bersih])
            label = str(mlp.predict(vek)[0])
            proba = pd.Series(mlp.predict_proba(vek)[0], index=[str(c) for c in mlp.classes_])
            st.markdown(f"### {EMOJI[label]} Sentimen: **{label.upper()}**")
            st.caption(f"Teks bersih: `{bersih}`")
            tabel = proba.reindex(URUTAN).rename("probabilitas").to_frame()
            st.dataframe(tabel.style.format("{:.2%}"), width="stretch")
            st.bar_chart(tabel)
