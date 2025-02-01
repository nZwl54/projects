# projects
import pandas as pd

df = pd.read_csv("veri.csv")
df.info()
df.head()
df2 = df.drop(["ulke","yil"], axis=1)
df2.describe()
import missingno as msno
msno.bar(df)
df.isnull().sum()
df.info()
df
df["enerji_yatırımlari_milyar_usd"] = df["enerji_yatırımlari_milyar_usd"].str.replace(r'\t', '', regex=True)
df["enerji_yatırımlari_milyar_usd"] = df["enerji_yatırımlari_milyar_usd"].str.replace(',', '.', regex=True).astype(float)
df.info()
df
import seaborn as sns
import matplotlib.pyplot as plt
df1 = df.drop(["ulke","yil"], axis=1)
correlation_matrix = df1.corr()

# Isı haritası ile korelasyonu görselleştirme
plt.figure(figsize=(8, 6))  # Grafik boyutunu ayarlama
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Grafik başlığı
plt.title("Korelasyon Isı Haritası")

# Grafiği gösterme
plt.show()

def plot_graph(x, column, ylabel, title, is_barh=False):
    """
    Genel grafik çizim fonksiyonu.
    """
    for i in x:
        data = df[df["ulke"] == i]

        # Veri kontrolü
        if data.empty:
            print(f"{i} için veri yok.")
            continue

        years = data["yil"]
        values = data[column]

        plt.figure()  # Yeni bir figür oluştur
        if is_barh:
            plt.barh(years, values)  # Horizontal bar grafiği
        else:
            plt.bar(years, values)  # Dikey bar grafiği

        plt.title(f"{i} {title}")
        plt.xlabel('Yıl')
        plt.ylabel(ylabel)
        plt.show()  # Grafiği göster

def grafikler(x):
    # Enerji Tüketimi Bar Grafiği
    plot_graph(x, "yillik_enerji_tuketimi_mtep", 'Enerji Tüketimi (Mtep)', "Yıllık Enerji Tüketimi")

    # Karbon Emisyonu Çizgi Grafiği
    for i in x:
        data = df[df["ulke"] == i]

        # Veri kontrolü
        if data.empty:
            print(f"{i} için veri yok.")
            continue

        years = data["yil"]
        emissions = data["yillik_karbon_emisyonu_milyon_ton_co2"]

        plt.figure()  # Yeni bir figür oluştur
        plt.plot(years, emissions, marker=".", markersize=5, markerfacecolor="red")
        plt.title(f"{i} Yıllık Karbon Emisyonu")
        plt.xlabel('Yıl')
        plt.ylabel('Karbon Emisyonu (Milyon Ton CO2)')
        plt.show()  # Grafiği göster

    # Toplu Taşıma Kullanımı Barh Grafiği
    plot_graph(x, "yillik_toplu_taşima_kullanımı_milyar", 'Toplu Taşıma Kullanımı (Milyar)', "Yıllık Toplu Taşıma Kullanımı", is_barh=True)

    # Hava Kalitesi AQI Bar Grafiği
    plot_graph(x, "hava_kalitesi_aqi", 'AQI', "Yıllık Hava Kalitesi (AQI)")

    # Enerji Yatırımları Bar Grafiği
    plot_graph(x, "enerji_yatırımlari_milyar_usd", 'Enerji Yatırımları (Milyar USD)', "Yıllık Enerji Yatırımları")

    # Kişi Başına Düşen Araç Bar Grafiği
    plot_graph(x, "kisi_basina_düsen_arac", 'Kişi Başına Düşen Araç', "Yıllık Kişi Başına Düşen Araç Grafiği")

# Listeyi vererek fonksiyonu çağırma
list1 = ["Endonezya", "Arjantin", "Japonya", "İtalya", "Türkiye"]
grafikler(list1)

def plot_averages(df, countries, columns):
    """
    Ülkelerin değişkenlerin ortalamasını hesaplayıp tek bir grafikte karşılaştırma.
    """
    # Ülkelerin ortalamalarını içeren bir DataFrame oluştur
    averages = []
    for country in countries:
        data = df[df["ulke"] == country]
        if data.empty:
            print(f"{country} için veri yok.")
            continue
        averages.append(data[columns].mean())

    # Ortalamaları bir DataFrame'e dönüştür
    averages_df = pd.DataFrame(averages, index=countries, columns=columns)

    # Grup sütun grafiği
    averages_df.plot(kind='bar', figsize=(12, 6), width=0.8)
    plt.title("Ülkelerin Tüm Yıllar İçin Değişkenlerin Ortalaması")
    plt.xlabel("Ülkeler")
    plt.ylabel("Ortalama Değerler")
    plt.xticks(rotation=45)
    plt.legend(title="Değişkenler", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Kullanım
columns = [
    "yillik_enerji_tuketimi_mtep",
    "yillik_karbon_emisyonu_milyon_ton_co2",
    "yillik_toplu_taşima_kullanımı_milyar",
    "hava_kalitesi_aqi",
    "enerji_yatırımlari_milyar_usd",
    "kisi_basina_düsen_arac"
]
plot_averages(df, list1, columns)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['ulke'] = le.fit_transform(df['ulke'])
df
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

X = df.drop(["yillik_toplu_taşima_kullanımı_milyar"], axis=1)
y = df["yillik_toplu_taşima_kullanımı_milyar"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Veriyi Ölçeklendir (Özellikle KNN ve Decision Tree için önemli)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Modelleri Kur ve Eğit

#  Lineer Regresyon Modeli
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

#  KNN (K-Nearest Neighbors) Modeli
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_scaled, y_train)

#  Random Forest Modeli
rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, y_train)

#  Decision Tree (Karar Ağacı) Modeli
dt_reg = DecisionTreeRegressor(random_state=42)
dt_reg.fit(X_train, y_train)

#  Modellerin Tahminleri
linear_pred = linear_reg.predict(X_test)
knn_pred = knn_reg.predict(X_test_scaled)
rf_pred = rf_reg.predict(X_test)
dt_pred = dt_reg.predict(X_test)



from sklearn.tree import export_graphviz
from IPython.display import Image
import pydot

# Karar Ağacını .dot formatında oluştur
dot_data = export_graphviz(
    dt_reg,
    out_file=None,  # None: Doğrudan bir string döner
    feature_names=X.columns,
    filled=True,
    rounded=True,
    special_characters=True
)

# Dot formatındaki veriyi görsele dönüştür
(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png("decision_tree.png")  # PNG dosyasını kaydet

# PNG dosyasını Colab hücresinde göster
Image("decision_tree.png")
# Random Forest'ten bir ağaç seçelim
selected_tree = rf_reg.estimators_[0]

# Karar ağacını .dot formatında oluştur
dot_data = export_graphviz(
    selected_tree,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    special_characters=True
)

# Dot formatındaki veriyi görsele dönüştür
(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png("random_forest_tree.png")  # PNG dosyasını kaydet

# Hücrede görüntüle
Image("random_forest_tree.png")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, linear_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Değerleri")
plt.title("Lineer Regresyon: Gerçek ve Tahmin Değerleri")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(knn_pred, y_test - knn_pred, alpha=0.6, color='purple')
plt.axhline(y=0, color='red', linestyle='--', label='0 Hata (Referans Çizgisi)')
plt.xlabel("Tahmin Edilen Değerler")
plt.ylabel("Hata (Gerçek - Tahmin)")
plt.title("KNN Modeli: Tahmin ve Hata İlişkisi (Scatter Plot)")
plt.legend()
plt.show()
#  Performans Değerlendirmesi
def evaluate_model_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

#  R² Skorları
models = ['Linear Regression', 'KNN', 'Random Forest', 'Decision Tree']
predictions = [linear_pred, knn_pred, rf_pred, dt_pred]
r2_scores = [evaluate_model_r2(y_test, pred) for pred in predictions]

# Performans Karşılaştırması (Grafik ile)
plt.figure(figsize=(8, 6))
plt.bar(models, r2_scores, color=['blue', 'green', 'orange', 'red'])
plt.xlabel('Model')
plt.ylabel('R-squared (R²)')
plt.title('Regresyon Modelleri Karşılaştırması (R² Skoru)')
plt.xticks(rotation=45)
plt.ylim(0, 1)  # R² değerinin 0 ile 1 arasında olacağını unutmayın
plt.show()

#  Performans Sonuçlarını Yazdır
for model, r2 in zip(models, r2_scores):
    print(f"{model} - R²: {r2:.4f}")
new_data = pd.DataFrame({
    "yil": [2022],
    "ulke": [2],
    "enerji_yatırımlari_milyar_usd": [40.0],
    "hava_kalitesi_aqi": [60],
    "kisi_basina_düsen_arac": [0.65],
    "yillik_enerji_tuketimi_mtep": [600],
    "yillik_karbon_emisyonu_milyon_ton_co2": [1300]
}, columns=X_train.columns)


new_data_scaled = scaler.transform(new_data)
new_data_scaled = scaler.transform(new_data)
# Lineer Regresyon Modeli ile Tahmin
linear_pred_new = linear_reg.predict(new_data)  # Yeni verilerle tahmin

# KNN Modeli ile Tahmin
knn_pred_new = knn_reg.predict(new_data_scaled)

# Random Forest Modeli ile Tahmin
rf_pred_new = rf_reg.predict(new_data)

# Decision Tree Modeli ile Tahmin
dt_pred_new = dt_reg.predict(new_data)

# Sonuçları Yazdır
print("Linear Regression Predictions:", linear_pred_new)
print("KNN Predictions:", knn_pred_new)
print("Random Forest Predictions:", rf_pred_new)
print("Decision Tree Predictions:", dt_pred_new)
# Japonya 2023 yılı toplu taşıma kullanımı 8,1 milyar
