import pandas as pd

print("Yerel veri okunuyor...\n")

# URL yerine doğrudan klasördeki dosyanın adını yazıyoruz
df = pd.read_csv("churn.csv")

print("Veri başarıyla yüklendi!\n")

print(f"Toplam Müşteri Sayısı: {df.shape[0]}")
print(f"Toplam Özellik (Sütun) Sayısı: {df.shape[1]}\n")

print("İlk 5 Müşteri Kaydı:")
print(df.head())