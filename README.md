# CLTV_PRED_RETAIL
Customer Life Time Value Prediction


![image](https://github.com/furkansukan/CLTV/assets/115731123/302eaf5e-177f-4ccc-a249-2ee93154c449)


Müşteri Ömrü Değeri (Customer Lifetime Value - CLTV), bir müşterinin belirli bir süre içinde bir şirket için ne kadar değer yaratacağını tahmin etmek için kullanılan bir yöntemdir.

CLTV yöntemi iki ana bölümden oluşur:

BG/NGD (Beta Geometrik/Negatif Binom Dağılımları) Alt Modeli: Koşullu beklenen işlem sayısının hesaplanması,
Gamma Gamma Alt Modeli: Koşullu beklenen ortalama karın hesaplanması.
CLTV modelinde BetaGeoFitter kullanılır.

Öngörü fonksiyonları haftalık olarak zamanlanırken, BetaGeoFitter fonksiyonunun zaman değişkeni aylık olarak hesaplanır.
