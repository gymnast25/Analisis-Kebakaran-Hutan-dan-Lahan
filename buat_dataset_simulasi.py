"""
Pembuat dataset SIMULASI tweet bertopik Kebakaran Hutan (karhutla).

Dipakai karena crawling langsung ke X/Twitter membutuhkan login/token dan tidak
bisa dijalankan di lingkungan pembuatan tugas ini. Format kolom sama persis
dengan hasil crawling: created_at, username, full_text.

Untuk data asli, jalankan crawling_twitter.py lalu timpa tweet_data.csv.
"""
import random
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)

lokasi = ["Riau", "Kalbar", "Kalteng", "Sumsel", "Jambi", "Pontianak", "Palangkaraya",
          "Pekanbaru", "Kalsel", "Ketapang", "Kubu Raya", "Banjarbaru", "Muaro Jambi"]
hashtags = ["#karhutla", "#kebakaranhutan", "#hutanterbakar", "#asapkarhutla", "#SiagaApi", ""]

positif = [
    "Alhamdulillah api di {l} berhasil padam setelah tim gabungan bekerja semalaman, warga merasa lega",
    "Hujan turun di {l}, titik api mulai padam dan kondisi membaik",
    "Terima kasih petugas damkar dan relawan, kebakaran lahan di {l} sudah terkendali",
    "Bantuan masker dan logistik sudah sampai di {l}, warga selamat dan aman di posko",
    "Satgas karhutla sigap memadamkan api di {l}, lahan mulai pulih",
    "Water bombing berhasil, api di {l} tertangani dan udara berangsur membaik",
    "Lega, kebakaran hutan di {l} akhirnya padam dan warga aman kembali ke rumah",
    "Kondisi {l} pulih setelah hujan deras, kualitas udara membaik dan situasi aman",
    "Apresiasi untuk tim pemadam, api di {l} terkendali dan tidak ada korban, semua selamat",
    "Bantuan dari pemerintah dan relawan terus mengalir ke {l}, warga merasa terbantu dan aman",
    "Patroli terpadu berhasil mencegah api meluas di {l}, hutan masih selamat",
    "Berkat kerja sigap petugas, karhutla di {l} tertangani cepat dan lahan bisa pulih",
]
negatif = [
    "Asap karhutla di {l} makin parah, jarak pandang menipis dan warga sesak napas",
    "Kebakaran hutan di {l} meluas, ratusan hektare lahan rusak dan hangus",
    "Kabut asap bahaya untuk anak-anak di {l}, banyak yang terkena ISPA",
    "Udara {l} sangat buruk, polusi asap parah dan sekolah terpaksa diliburkan",
    "Api terus meluas di {l}, hutan rusak parah dan satwa kehilangan habitat",
    "Warga {l} mengungsi karena asap pekat, kondisi sangat berbahaya",
    "Penerbangan di {l} dibatalkan karena kabut asap, kerugian ekonomi parah",
    "Lahan gambut di {l} terus terbakar, asap makin pekat dan polusi makin parah",
    "Korban ISPA bertambah di {l}, rumah sakit penuh dan situasi sangat bahaya",
    "Hutan {l} ludes terbakar, kerusakan ekosistem parah dan sulit pulih",
    "Pemadaman lambat, api di {l} meluas dan asap bikin sesak seisi kota",
    "Sudah seminggu asap menyelimuti {l}, udara bahaya dan warga mengeluh",
]
netral = [
    "BMKG mengumumkan jumlah titik panas di {l} hari ini",
    "Rapat koordinasi penanganan karhutla digelar di kantor gubernur {l}",
    "Pemerintah daerah {l} menetapkan status siaga darurat karhutla",
    "Berita terbaru soal kebakaran lahan di {l} bisa dibaca di portal resmi",
    "Tim patroli karhutla melakukan pemantauan titik api di wilayah {l}",
    "Sosialisasi larangan membakar lahan dilakukan kepada petani di {l}",
    "Data hotspot satelit untuk {l} diperbarui setiap hari",
    "Kepala BPBD {l} memberikan keterangan pers terkait karhutla",
    "Posko karhutla dibuka di {l} sejak pagi tadi",
    "Pesawat patroli udara berangkat dari bandara {l} untuk pemantauan",
]
campuran = [  # sengaja memuat kata positif & negatif sekaligus (menguji kasus seimbang/dominan)
    "Asap masih parah di {l} tapi bantuan masker sudah datang",
    "Api di {l} meluas, namun petugas sigap dan warga selamat",
    "Kabut asap bahaya, untungnya warga {l} aman di posko",
    "Hutan {l} rusak, semoga lekas pulih dan terkendali",
]
akun = ["budi", "sari", "agus", "dewi", "rina", "andi", "joko", "putri", "fajar", "lisa",
        "dimas", "nina", "rudi", "tono", "maya", "eko", "wawan", "yuni", "hadi", "citra"]


def bumbui(teks):
    r = random.random()
    if r < 0.25:
        teks += " https://t.co/" + "".join(random.choices("abcdefghXYZ123456789", k=8))
    if r > 0.7:
        teks = "@" + random.choice(["BNPB_Indonesia", "KLHK_RI", "BMKG", "detikcom", "kompascom"]) + " " + teks
    h = random.choice(hashtags)
    if h:
        teks += " " + h
    if random.random() < 0.3:
        teks += " " + str(random.randint(1, 900)) + " titik" if random.random() < 0.5 else " !!!"
    if random.random() < 0.2:
        teks = teks.upper() if random.random() < 0.3 else teks.capitalize()
    return teks


rows = []
mulai = datetime(2026, 8, 1, 6, 0)
komposisi = [(positif, 120), (negatif, 160), (netral, 90), (campuran, 30)]
for kumpulan, n in komposisi:
    for _ in range(n):
        t = random.choice(kumpulan).format(l=random.choice(lokasi))
        waktu = mulai + timedelta(minutes=random.randint(0, 60 * 24 * 85))
        rows.append({
            "created_at": waktu.strftime("%Y-%m-%d %H:%M:%S"),
            "username": random.choice(akun) + str(random.randint(1, 999)),
            "full_text": bumbui(t),
        })

df = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
df = df.sort_values("created_at").reset_index(drop=True)
df.to_csv("tweet_data.csv", index=False, encoding="utf-8")
print("Tersimpan tweet_data.csv:", df.shape)
