"""
Crawling tweet asli (Langkah 1) memakai tweet-harvest.

Prasyarat:
  1. Node.js terpasang (https://nodejs.org)
  2. Punya auth_token akun X/Twitter (cookie `auth_token` dari browser yang login)
Jalankan:
  python crawling_twitter.py <AUTH_TOKEN>

Hasil: tweets-data/tweet_data_raw.csv -> dikonversi menjadi tweet_data.csv
dengan kolom created_at, username, full_text.
"""
import subprocess
import sys
import pandas as pd

if len(sys.argv) < 2:
    sys.exit("Pemakaian: python crawling_twitter.py <AUTH_TOKEN>")

token = sys.argv[1]
kueri = "kebakaran hutan OR karhutla lang:id"

subprocess.run(
    ["npx", "-y", "tweet-harvest@latest", "-o", "tweet_data_raw.csv",
     "-s", kueri, "-l", "400", "--token", token],
    check=True,
)

df = pd.read_csv("tweets-data/tweet_data_raw.csv")
df = df[["created_at", "username", "full_text"]].dropna().drop_duplicates("full_text")
df.to_csv("tweet_data.csv", index=False, encoding="utf-8")
print("Tersimpan tweet_data.csv:", df.shape)
