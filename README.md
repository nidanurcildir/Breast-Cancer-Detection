🧠 AI-Powered Breast Cancer Diagnosis System
🌟 Proje Özeti
Meme kanseri teşhisi için geliştirilmiş makine öğrenimi modeli. Wisconsin veri seti üzerinde %97 doğruluk oranıyla çalışan lojistik regresyon tabanlı akıllı tanı sistemi.

📊 Temel Özellikler
🏥 Klinik Destek: Hekimlere yardımcı tanı aracı

⚡ Hızlı Analiz: Anında tümör sınıflandırması

📈 Yüksek Doğruluk: 30 özellikle optimize edilmiş model

⚖️ Dengeli Tahmin: Sınıf ağırlıklandırmasıyla güvenilirlik

🛠 Teknik Detaylar
python
# Örnek Kod Snippet
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)  # 97% accuracy
📌 Kullanılan Teknolojiler
Kategori	Teknolojiler
🐍 Dil	Python 3.11
📊 Veri Bilimi	pandas, numpy
🤖 ML	scikit-learn
📈 Görselleştirme	matplotlib, seaborn
🚀 Başlarken
bash
git clone https://github.com/[kullanıcı-adı]/meme-kanseri-teshisi.git
cd meme-kanseri-teshisi
pip install -r requirements.txt
📂 Veri Seti Özellikleri
569 hasta kaydı (212 malign, 357 benign)

30 radyolojik özellik

Tamamen anonimleştirilmiş veriler
