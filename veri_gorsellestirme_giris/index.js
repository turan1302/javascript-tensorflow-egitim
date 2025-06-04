/*
🔢 1. TensorFlow Grafiği Nedir?
TensorFlow’da “grafik (graph)” kelimesi iki anlama gelebilir:

📌 Anlam 1: Hesaplama Grafiği (Computational Graph)
Bu grafik; bir modeldeki tüm matematiksel işlemlerin (çarpma, toplama, aktivasyon, vs.) düğümler (nodes) ve bağlantılar (edges) ile gösterildiği arka plan yapıdır.

Örn: y = x * 2 + 1 fonksiyonu için:

x = girdi

x * 2 = bir düğüm (çarpma)

+1 = başka düğüm (toplama)
Bunların hepsi bir “hesaplama grafiği” içinde tutulur.

TensorFlow bu şekilde işlemleri optimize eder ve çalıştırır.

Ama biz genelde bunun iç yapısıyla ilgilenmeyiz.

📌 Anlam 2: Eğitim Sürecinde Ortaya Çıkan Verilerin Grafiği
İşte bizim için önemli olan bu:

Kayıp değeri (loss) zamanla nasıl düşüyor?

Doğruluk (accuracy) zamanla nasıl artıyor?

Modelin yaptığı tahminler ne kadar yakın?

🎯 Bu verileri grafik halinde çizmek, modeli anlamamıza çok yardımcı olur.

🔍 2. Hangi Verileri Çizeriz?
Modeli eğitirken şu verileri grafiğe dökeriz:

Ne?	Açıklama
Epoch	Eğitim turu sayısı (örneğin 0-500 arası)
Loss (kayıp)	Gerçek ve tahmin farkı (ne kadar yanlış?)
Accuracy	Ne kadar doğru tahmin ediyor
Tahminler	Gerçek verilere göre model ne diyor?

📊 3. Neden Grafik Gerekli?
Grafik çizmek bize şunu sağlar:

Model öğreniyor mu, yoksa ezberliyor mu?

Fazla mı karmaşık, yoksa basit mi?

Kaç epoch sonrası öğrenme duruyor?

Yani sezgisel olarak anlayabiliriz ki “tamam bu model artık yeterince iyi” veya “hala sorun var”.
 */

const tf = require("@tensorflow/tfjs-node")
const {plot} = require("nodeplotlib");

// Eğitim verisi: x → y = 2x + 1
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([3, 5, 7, 9], [4, 1]);

async function trainModel(){
    const model = tf.sequential()
    model.add(tf.layers.dense({
        units : 1,
        inputShape : [1]
    }))

    model.compile({
        optimizer : "adam",
        loss : "meanSquaredError"
    })

    const lossValues = []

    // Eğitimi başlat
    await model.fit(xs,ys,{
        epochs : 100,
        callbacks : {
            onEpochEnd : async (epoch, logs) => {
                lossValues.push({
                    x : epoch,
                    y : logs.loss
                })
            }
        }
    })

    plot([{
        x : lossValues.map(p=>p.x),
        y : lossValues.map(p=>p.y),
        type : "line",
        name : "Kayıp (Loss)"
    }])
}

trainModel()
