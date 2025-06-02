/*
Süper kanka, o zaman önce **hayatla bağlantılı**, sade bir şekilde teorik anlatımı yapalım:
(Bu işin temelini anlamadan pratikte değişiklik yapmak kuru ezber olur.)

---

## 🔌 Aktivasyon Fonksiyonları Nedir?

Bir yapay sinir ağı katmanındaki **nöronlar**, girdilere karşılık bazı sayılar hesaplar (bunlara "ağırlıklı toplam" denir).
Ama bu **çıktılar her zaman işe yaramaz**. Çünkü:

* Negatif değerler olabilir
* Sonsuz büyüklükte sayılar olabilir
* Doğrusal (linear) olabilir → Karmaşık şeyleri öğrenemez

İşte bu yüzden araya **aktivasyon fonksiyonu** girer.

---

## 🧠 Hayattan Örnekle Anlatım:

### 📍Durum:

Bir arkadaşına 5 farklı teklif yapıyorsun (film, gezi, yemek, oyun, vs).
Onun bunlara olan ilgisi + geçmiş deneyimleriyle bir karar alıyor.

Şimdi...

* Arkadaşının **zihinsel kararı** → yapay sinir ağına benzer.
* **Her teklif için ayrı etki** → ağırlıklar (weights)
* Sonuç: "Yapalım mı, yapmayalım mı?" → **aktivasyon fonksiyonu** gibi davranır.

---

## 🧪 Peki, Aktivasyon Ne Yapar?

| Amaç                         | Açıklama                                                                  |
| ---------------------------- | ------------------------------------------------------------------------- |
| **Karar sınırı koyar**       | “Bu değer 0’ın altındaysa boş ver” der (ReLU gibi)                        |
| **İhtimali ölçer**           | "Bunu %80 yaparız, %20 yapmayız" der (Sigmoid gibi)                       |
| **Pozitif/negatif yorumlar** | “Bu durum -1 kadar olumsuz, +1 kadar olumlu” der (Tanh gibi)              |
| **Hiç dokunmaz**             | “Direkt çıktıyı ver” der (Linear gibi – genellikle çıkış katmanında olur) |

---

## 🔍 Aktivasyon Fonksiyonlarına Teorik Bakış:

### 1. **ReLU (Rectified Linear Unit)**

* En çok kullanılan.
* Negatifleri 0 yapar, pozitifi aynen bırakır.
* **Hayat örneği:** "Canım istemezse hiç yapmam, ama istersem tam yaparım."

```math
f(x) = max(0, x)
```

---

### 2. **Sigmoid**

* Çıktıyı 0 ile 1 arasına sıkıştırır.
* Genellikle sınıflandırmada (binary) kullanılır.
* **Hayat örneği:** "Bu fikre %85 katılıyorum."

```math
f(x) = 1 / (1 + e^(-x))
```

---

### 3. **Tanh (Hyperbolic Tangent)**

* Çıktı -1 ile 1 arasında olur.
* Negatif ve pozitif yorumları birlikte verir.
* **Hayat örneği:** "Bu durum -1 kadar olumsuz, 0.5 kadar olumlu" gibi düşün.

```math
f(x) = (e^x - e^-x) / (e^x + e^-x)
```

---

### 4. **Linear**

* Çıktıya hiç dokunmaz.
* Genellikle **regresyon çıkışlarında** kullanılır.
* **Hayat örneği:** "Ne hissediyorsam onu söylüyorum."

```math
f(x) = x
```

---

## 🧭 Hangi Durumda Hangisi?

| Senaryo                               | Aktivasyon              |
| ------------------------------------- | ----------------------- |
| Görsel, metin tanıma, regresyon       | ReLU                    |
| Olasılık / sınıflandırma (evet-hayır) | Sigmoid                 |
| Duygu gibi + ve - ayrımı olan işler   | Tanh                    |
| Sayı tahmini / regresyon çıkışı       | Linear (sadece çıkışta) |

---

## ⚠️ Not:

Hiç aktivasyon fonksiyonu koymazsan:

* Bütün katmanlar **doğrusal dönüşüm** olur.
* Ne kadar katman eklersen ekle, **tek satırlık denklem** gibi çalışır.
* Bu durumda sinir ağın **"derin" değil, "düz çizgi" olur.**

---

Eğer istersen şimdi bu öğrendiklerimizi **uygulamaya** dökelim:

> Aynı veriyle 3 model yapalım ve karşılaştıralım:
>
> * ReLU modeli
> * Sigmoid modeli
> * Tanh modeli

Hazırsan kod kısmına geçelim mi kanka?

 */
