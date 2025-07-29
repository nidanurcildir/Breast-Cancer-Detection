# 1. Gerekli kütüphaneleri yükle
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

# 2. Veri setini yükle
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)  # 0: Kötü huylu, 1: İyi huylu

# 3. Veriyi eğitim/test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Veriyi ölçeklendir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Modeli eğit (Lojistik Regresyon)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 6. Tahmin yap ve performansı ölç
y_pred = model.predict(X_test_scaled)
print("\n=== TEMEL MODEL PERFORMANSI ===")
print("Doğruluk (Accuracy):", accuracy_score(y_test, y_pred))
print("Karmaşıklık Matrisi:\n", confusion_matrix(y_test, y_pred))

# 7. Cross-Validation ile Model Stabilitesi
print("\n=== CROSS-VALIDATION SONUÇLARI ===")
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Ortalama Doğruluk: {np.mean(cv_scores):.4f}")
print(f"Standart Sapma: {np.std(cv_scores):.4f}")
print("Tüm Katlar:", cv_scores)

# 8. Özellik Önem Analizi
print("\n=== ÖZELLİK ÖNEM SIRALAMASI ===")
importances = pd.DataFrame({
    'Feature': cancer.feature_names,
    'Importance': np.abs(model.coef_[0])
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10,6))
plt.barh(importances['Feature'][:10], importances['Importance'][:10])
plt.title('En Önemli 10 Özellik')
plt.xlabel('Katsayı Büyüklüğü')
plt.show()

# 9. Sınıf Dengesizliği Düzeltme
print("\n=== SINIF DENGESİZLİĞİ DÜZELTME ===")
weights = class_weight.compute_sample_weight('balanced', y_train)
balanced_model = LogisticRegression(max_iter=1000)
balanced_model.fit(X_train_scaled, y_train, sample_weight=weights)

y_pred_balanced = balanced_model.predict(X_test_scaled)
print("Dengelemeli Model Doğruluk:", accuracy_score(y_test, y_pred_balanced))
print("Yeni Karmaşıklık Matrisi:\n", confusion_matrix(y_test, y_pred_balanced))

# 10. Sınıf Dağılımı Görselleştirme
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.bar(['Kötü Huylu', 'İyi Huylu'], [sum(y == 0), sum(y == 1)], color=['red', 'green'])
plt.title('Veri Setindeki Sınıf Dağılımı')

plt.subplot(1,2,2)
plt.bar(['Kötü Huylu', 'İyi Huylu'],
        [confusion_matrix(y_test, y_pred_balanced)[0,0],
        confusion_matrix(y_test, y_pred_balanced)[1,1]],
        color=['red', 'green'])
plt.title('Dengelemeli Model Tahminleri')
plt.show()