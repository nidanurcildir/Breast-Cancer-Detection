# 🩺 AI-Powered Breast Cancer Diagnosis

## 📌 Proje Hakkında
**Wisconsin Meme Kanseri Veri Seti** kullanılarak geliştirilen makine öğrenimi modeli. Lojistik regresyon ile %97 doğruluk oranına ulaşılmıştır.

```python
# Örnek Kod
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)  # 97% accuracy

🛠 Teknoloji Yığını
Kategori	Teknolojiler
Programlama Dili	Python 3.11
Veri İşleme	pandas, numpy
Makine Öğrenimi	scikit-learn
Görselleştirme	matplotlib, seaborn

🚀 Kurulum
git clone https://github.com/nidanurcildir/Breast-Cancer-Detection.git
cd Breast-Cancer-Detection
pip install -r requirements.txt

📊 Veri Seti
Toplam Örnek: 569 (212 kötü huylu, 357 iyi huylu)
Özellik Sayısı: 30 radyolojik parametre
Veri Kaynağı: Scikit-learn gömülü veri seti

✨ Öne Çıkanlar
Sınıf dengesizliği optimizasyonu
Cross-validation ile model stabilitesi
Özellik önem analizi
Kapsamlı görselleştirmeler
