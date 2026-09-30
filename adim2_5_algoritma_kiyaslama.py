import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

df = pd.read_csv("churn_temiz.csv")
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modeller = {
    "Lojistik Regresyon": LogisticRegression(random_state=42),
    "Karar Ağacı (Decision Tree)": DecisionTreeClassifier(random_state=42),
    "Destek Vektör (SVM)": SVC(random_state=42),
    "Rastgele Orman (Random Forest)": RandomForestClassifier(class_weight='balanced', random_state=42)
}

print("Algoritmalar yarışıyor, lütfen bekleyin...\n")

# YENİ: Sonuçları tutacağımız boş liste
sonuclar_listesi = []

for isim, model in modeller.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred) * 100 
    
    # Her modelin sonucunu listeye ekliyoruz
    sonuclar_listesi.append({
        'Algoritma': isim,
        'Genel Başarı (Accuracy) %': accuracy,
        'Kayıp Yakalama (Recall) %': recall
    })
    
    print(f"{isim:<30} | Acc: %{accuracy:<5.2f} | Recall: %{recall:.2f}")

# YENİ: Listeyi DataFrame yapıp CSV olarak dışa aktarıyoruz
sonuclar_df = pd.DataFrame(sonuclar_listesi)
sonuclar_df.to_csv("model_metrikleri.csv", index=False)

print("\n✅ YARIŞMA BİTTİ! Gerçek test sonuçları 'model_metrikleri.csv' dosyasına dinamik olarak kaydedildi.")
print("Streamlit arayüzü artık grafiği bu dosyadan canlı olarak çekecek.")