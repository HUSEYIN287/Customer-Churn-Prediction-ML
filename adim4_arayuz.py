import streamlit as st
import pandas as pd
import joblib
import io

# --- YARDIMCI FONKSİYON: KİŞİSELLEŞTİRİLMİŞ KAMPANYA MOTORU (GELİŞTİRİLMİŞ) ---
def kampanya_belirle(row):
    """Müşterinin özelliklerine bakarak en uygun kampanyayı seçer."""
    aylik_fatura = row.get('MonthlyCharges', 0)
    toplam_fatura = row.get('TotalCharges', 0)
    kalma_suresi = row.get('tenure', 0)
    kontratsiz = (row.get('Contract_One year', 0) == 0 and row.get('Contract_Two year', 0) == 0)
    fiber_kullanicisi = (row.get('InternetService_Fiber optic', 0) == 1)
    yasli_vatandas = (row.get('SeniorCitizen', 0) == 1)
    
    if kalma_suresi > 36 and toplam_fatura > 3000 and kontratsiz:
        return "VIP Alarm: Sadık müşteri kaybediliyor! Özel müşteri temsilcisi atanmalı, şartsız %30 indirim ve ücretsiz cihaz yenileme sunulmalı."
    elif kalma_suresi < 12 and aylik_fatura > 90:
        return "Fatura Şoku: Müşteri fiyatı yüksek bulmuş olabilir. İlk 6 ay %25 sabit fiyat garantisi veya bir alt hız paketine ücretsiz geçiş teklif edin."
    elif fiber_kullanicisi and aylik_fatura > 70 and kontratsiz:
        return "Altyapı/Hizmet Riski: Fiber bağlantı kalitesini sorgulayın. Ücretsiz premium modem veya 3 ay ücretsiz TV+ / Sinema paketi hediye edin."
    elif yasli_vatandas and aylik_fatura > 50:
        return "Demografik Risk: Emekli müşteri özel paketi sunun. İhtiyacı olmayan ek servisleri kapatarak faturayı düşürmeyi teklif edin."
    elif kontratsiz and kalma_suresi > 12:
        return "Taahhüt Boşluğu: 12 aylık yeni taahhüt karşılığında ilk 2 ay ücretsiz kullanım ve sürpriz gigabayt hediyesi sunun."
    elif kalma_suresi < 6:
        return "Erken Kayıp: Sisteme alışamamış olabilir. Memnuniyet anketi araması yapın, ilk faturasını şirket olarak karşılamayı teklif edin."
    else:
        return "Genel Risk: %15 sadakat indirimi veya bir sonraki faturada 100 TL iade uygulayarak elde tutmaya çalışın."
# --------------------------------------------------------------

# 1. Eğittiğimiz modeli, ölçeklendiriciyi ve temizlenmiş veriyi yüklüyoruz
model = joblib.load('churn_modeli.pkl')
scaler = joblib.load('scaler.pkl')
df = pd.read_csv('churn_temiz.csv')

# 2. Sayfa tasarımı ve başlıklar
st.set_page_config(page_title="Churn Analiz Paneli", page_icon="📊", layout="wide")
st.title("📊 Müşteri Kayıp (Churn) Analiz Paneli")
st.markdown("Bu panel, makine öğrenmesi modeli kullanarak müşterilerin şirketi terk etme riskini hesaplar.")
st.divider()

# --- 1. BÖLÜM: BİREYSEL MÜŞTERİ ANALİZİ ---
st.header("👤 1. Bireysel Müşteri Seçimi ve Analizi")

if "slider_degeri" not in st.session_state:
    st.session_state.slider_degeri = 0
if "kutu_degeri" not in st.session_state:
    st.session_state.kutu_degeri = 0

def slider_degistiginde():
    st.session_state.kutu_degeri = st.session_state.slider_degeri

def kutu_degistiginde():
    st.session_state.slider_degeri = st.session_state.kutu_degeri

col1, col2 = st.columns(2)
with col1:
    st.slider("Kaydırarak Seçin:", 0, len(df)-1, key="slider_degeri", on_change=slider_degistiginde)
with col2:
    st.number_input("Veya Manuel Girin:", min_value=0, max_value=len(df)-1, step=1, key="kutu_degeri", on_change=kutu_degistiginde)

musteri_id = st.session_state.slider_degeri
secilen_musteri = df.iloc[[musteri_id]]

X_yeni = secilen_musteri.drop('Churn', axis=1)
gercek_durum = secilen_musteri['Churn'].values[0]

st.write("Seçilen Müşterinin Verileri:")
st.dataframe(X_yeni)

if st.button("🤖 Yapay Zeka Risk Analizi Yap"):
    X_yeni_scaled = scaler.transform(X_yeni)
    tahmin = model.predict(X_yeni_scaled)
    olasilik = model.predict_proba(X_yeni_scaled)[0][1] 
    
    st.divider()
    st.subheader("Makine Öğrenmesi Sonucu:")
    
    if tahmin[0] == 1:
        st.error(f"🚨 DİKKAT! Bu müşterinin ayrılma riski YÜKSEK! (Risk Oranı: %{olasilik*100:.1f})")
        aylik_fatura = secilen_musteri['MonthlyCharges'].values[0]
        yillik_zarar = aylik_fatura * 12
        st.warning(f"💸 **Potansiyel Finansal Kayıp:** Bu müşteri ayrılırsa şirketin yıllık tahmini ciro kaybı **{yillik_zarar:.2f} TL** olacaktır.")
        ozel_kampanya = kampanya_belirle(secilen_musteri.iloc[0])
        st.info(f"💡 **Yapay Zeka Çözüm Önerisi:** {ozel_kampanya}")
    else:
        st.success(f"✅ GÜVENLİ! Bu müşterinin ayrılma riski DÜŞÜK. (Risk Oranı: %{olasilik*100:.1f})")
        
    st.info(f"Veritabanındaki Gerçek Durumu: {'Ayrılmış (1)' if gercek_durum == 1 else 'Ayrılmamış, Aktif (0)'}")

    st.divider()
    st.subheader("💡 Modelin Karar Kriterleri (Genel)")
    st.markdown("Yapay zeka genel olarak müşteri kaçışlarını tahmin ederken en çok aşağıdaki 5 özelliğe dikkat etmektedir:")
    onem_dereceleri = pd.DataFrame({
        'Özellik': X_yeni.columns,
        'Önem Derecesi (%)': model.feature_importances_ * 100
    })
    grafik_verisi = onem_dereceleri.sort_values(by='Önem Derecesi (%)', ascending=False).head(5).set_index('Özellik')
    st.bar_chart(grafik_verisi)

# --- YENİ GÜNCELLENEN KISIM: YÖNETİCİ RAPORU VE DİNAMİK GRAFİKLER ---
st.divider()
st.header("📈 2. Yönetici Raporları ve Performans Analizi")

# Analiz Sekmeleri (Tabs)
tab1, tab2 = st.tabs(["Dinamik Özellik Analizi (Müşteriler)", "Yapay Zeka Algoritma Karşılaştırması"])

with tab1:
    st.markdown("Aşağıdaki listeden herhangi bir özelliği seçerek, o özelliğe sahip müşterilerin şirketi terk etme (Churn) oranlarını inceleyebilirsiniz.")
    
    # Churn hariç tüm sütunları açılır listeye (selectbox) alıyoruz
    analiz_sutunlari = df.drop('Churn', axis=1).columns
    secilen_sutun = st.selectbox("Grafiğini görmek istediğiniz özelliği seçin:", analiz_sutunlari)
    
    # Seçilen sütuna göre veriyi gruplayıp, Churn oranını hesaplıyoruz
    churn_analiz_df = df.groupby(secilen_sutun)['Churn'].mean() * 100
    
    st.write(f"**{secilen_sutun}** özelliğine göre Müşteri Kaçış Oranları (%)")
    st.bar_chart(churn_analiz_df)

with tab2:
    st.markdown("Projenin geliştirilme sürecinde farklı sınıflandırma algoritmaları test edilmiştir. Churn projelerinde asıl hedef **Ayrılacak Müşterileri Yakalama (Recall)** başarısı olduğu için nihai model olarak en iyi performansı gösteren **Random Forest** seçilmiştir.")
    
    try:
        # Rakamları elle yazmak yerine arka planda yapılan testin dosyasını okuyoruz (DİNAMİK YAPI)
        algoritma_sonuclari = pd.read_csv("model_metrikleri.csv")
        algoritma_sonuclari = algoritma_sonuclari.set_index('Algoritma')
        
        # Sonuçları çizdiriyoruz
        st.bar_chart(algoritma_sonuclari['Kayıp Yakalama (Recall) %'])
        
        # İsteyen yönetici için tablonun orijinal halini de basıyoruz
        st.dataframe(algoritma_sonuclari)
        
    except FileNotFoundError:
        st.warning("⚠️ Algoritma karşılaştırma verisi bulunamadı. Lütfen arka planda önce algoritma test scriptini (adim2_5_algoritma_kiyaslama.py) çalıştırın.")
# ------------------------------------------------

# --- 3. BÖLÜM: TOPLU ANALİZ VE İNDİRME ---
st.divider()
st.header("📂 3. Toplu Veritabanı Analizi")
st.markdown("Tüm veritabanını tarayarak riskli müşterileri ve onlara özel kampanya önerilerini içeren listeyi indirebilirsiniz.")

if st.button("Tüm Veritabanını Tara ve Raporu İndir"):
    X_tum = df.drop('Churn', axis=1)
    X_tum_scaled = scaler.transform(X_tum)
    tum_tahminler = model.predict(X_tum_scaled)
    
    riskli_musteriler_df = df[tum_tahminler == 1].copy()
    riskli_musteriler_df['Önerilen_Kampanya'] = riskli_musteriler_df.apply(kampanya_belirle, axis=1)
    
    st.success(f"✅ Tarama Tamamlandı! {len(riskli_musteriler_df)} riskli müşteri tespit edildi.")
    
    riskli_musteriler_df.index.name = 'Musteri_Veritabani_No'
    riskli_musteriler_df = riskli_musteriler_df.reset_index()
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        riskli_musteriler_df.to_excel(writer, index=False, sheet_name='Riskli Müşteriler')
    
    excel_veri = buffer.getvalue()
    
    st.download_button(
        label="⬇️ Riskli Müşteriler Listesini İndir (Excel .xlsx)",
        data=excel_veri,
        file_name='riskli_musteriler_kampanya_raporu.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )