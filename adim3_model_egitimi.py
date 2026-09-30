import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier # YENİ: SVM yerine Random Forest geldi
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Temizlenmiş veriyi okuyoruz
df = pd.read_csv("churn_temiz.csv")

# 2. X (Özellikler) ve y (Hedef / Churn) olarak ayırıyoruz
X = df.drop('Churn', axis=1)
y = df['Churn']

# 3. Veriyi %80 Eğitim ve %20 Test olarak bölüyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Modeli beslemeden önce verileri ölçeklendiriyoruz (Standardizasyon)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Yapay Zeka (Random Forest) eğitiliyor, lütfen bekleyin...\n")

# 5. Random Forest modelini oluşturuyor ve eğitiyoruz
# YENİ: class_weight='balanced' parametresi ile yapay zekaya "kaçanlara daha çok odaklan" diyoruz
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Test verisi ile modelin performansını ölçüyoruz
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Başarısı (Accuracy): %{accuracy * 100:.2f}\n")
print("Detaylı Performans Raporu:")
print(classification_report(y_test, y_pred))

# --- YENİ EKLENEN KISIM: ÖZELLİK ÖNEMİ (FEATURE IMPORTANCE) ---
print("-" * 50)
print("🎯 Müşteri Kaçışında En Etkili 5 Özellik:")

# Özelliklerin önem derecelerini alıp bir tabloya çeviriyoruz
feature_importances = pd.DataFrame({
    'Özellik': X.columns,
    'Önem Derecesi (%)': model.feature_importances_ * 100
})

# En yüksekten en düşüğe sıralayıp ilk 5'i ekrana basıyoruz
en_etkili_5 = feature_importances.sort_values(by='Önem Derecesi (%)', ascending=False).head(5)
print(en_etkili_5.to_string(index=False))
print("-" * 50)
# -------------------------------------------------------------

# 7. Eğitilen modeli ve ölçeklendiriciyi (scaler) dosya olarak kaydediyoruz
# Kayıt isimlerini aynı bıraktık ki arayüz (Streamlit) kodumuz bozulmasın
joblib.dump(model, 'churn_modeli.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\nYeni Random Forest Modeli ve Scaler başarıyla kaydedildi! (churn_modeli.pkl ve scaler.pkl)")