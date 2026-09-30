import pandas as pd

# 1. Yerel verimizi okuyoruz
df = pd.read_csv("churn.csv")
print("Orijinal Veri Boyutu:", df.shape)

# 2. Modelin kafasını karıştıracak olan Müşteri ID'sini siliyoruz
df.drop('customerID', axis=1, inplace=True)

# 3. TotalCharges (Toplam Fatura) metin olarak kalmış, onu ondalıklı sayıya çeviriyoruz
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0) # Boş kalanları 0 yapıyoruz

# 4. Hedef değişkenimiz olan Churn (Terk etti mi?) sütununu 1 ve 0 yapıyoruz
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 5. Kalan tüm metin sütunlarını (Cinsiyet, İnternet Tipi vs.) One-Hot Encoding ile sayısallaştırıyoruz
# dtype=int diyerek True/False yerine 1/0 yazmasını sağlıyoruz
df = pd.get_dummies(df, drop_first=True, dtype=int)

print("\nVeri başarıyla temizlendi ve sayısallaştırıldı!")
print("Yeni Veri Boyutu:", df.shape)

# 6. Temizlenmiş veriyi sonraki adımlar için yeni bir CSV olarak kaydediyoruz
df.to_csv("churn_temiz.csv", index=False)
print("\nTemizlenmiş veri 'churn_temiz.csv' adıyla klasöre kaydedildi.")