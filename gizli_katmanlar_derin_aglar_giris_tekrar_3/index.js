/*
📘 Görev:
Girdi:
[haftanın günü (0=pazar, 1=pazartesi...), sıcaklık (°C), yağmur var mı? (0/1), tatil günü mü? (0/1)]

Çıktı:
[müşteri sayısı, günlük gelir ₺]
 */

const tf = require("@tensorflow/tfjs-node")

// Girdi: [haftanın günü (0=pazar), sıcaklık, yağmur var mı? (0/1), tatil mi? (0/1)]
const xsRaw = [
    [0, 20, 1, 1],
    [1, 25, 0, 0],
    [2, 22, 1, 0],
    [3, 27, 0, 0],
    [4, 30, 0, 0],
    [5, 28, 0, 0],
    [6, 24, 1, 1],
    [0, 21, 1, 1],
    [2, 26, 0, 0],
    [5, 29, 0, 0]
]

// Çıktı: [müşteri sayısı, günlük gelir ₺]
const ysRaw = [
    [120, 6000],
    [200, 10000],
    [150, 7500],
    [220, 11000],
    [250, 12500],
    [270, 13500],
    [130, 6500],
    [110, 5500],
    [210, 10500],
    [260, 13000]
]


// normalizasyon yazalım
function normalize(data){
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normalize = dataT.sub(min).div(max.sub(min)) // normalizassyon fonksiyonu
    return {normalize,min,max}
}

// normalize yapalım
const {normalize : xsNorm, min : xsMin, max : xsMax} = normalize(xsRaw)
const {normalize : ysNorm, min : ysMin, max : ysMax} = normalize(ysRaw)

// model oluşturalım
const model = tf.sequential()

// giriş katmanı
model.add(tf.layers.dense({
    units : 10, // 10 nöron
    inputShape : [4], // 4 giriş
    activation : "relu", // karmaşık fonksiyon çözecek
    kernelRegularizer : tf.regularizers.l2({
        l2 : 0.01
    })
}))

// dropout kısmı (gizli katman)
model.add(tf.layers.dropout({
    rate : 0.3
}))

// gizli katman
model.add(tf.layers.dense({
    units : 6, // 6 nöron
    activation : "relu" // karmaşık fonksiyon çözecek
}))

// çıkış katmanı
model.add(tf.layers.dense({
    units : 2, // 2 nöron
    activation : "linear" // doğrudan çıktı
}))

// model derle
model.compile({
    optimizer : "adam", // uyarlanabilir model tahmini. Ağırlıkları en doğru şekilde güncellemeye çalışır
    loss : "meanSquaredError"
})

// model eğit ve tahmöin yap
async function trainAndPredict(){
    const fit_result = await model.fit(xsNorm,ysNorm,{
        epochs : 400
    })

    await model.save("file://./model")

    const loadedModel = await tf.loadLayersModel("file://./model/model.json")

    const inputRaw = tf.tensor2d([[5, 28, 0, 0]]) // Cuma, 28°C, yağmur yok, tatil değil
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))

    const predNorm = await loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)

    pred.print()

    console.log("Öğrenme Kaybı: ",fit_result.history.loss)

    const trust_value =  await model.evaluate(xsNorm,ysNorm)
    console.log("Doğruluk Analizi: ",trust_value.dataSync()[0])

    /*
    🧠 Beklenen Tahmin Aralığı:
    Müşteri Sayısı: 250 – 270
    Günlük Gelir: 12500 – 13500 ₺
     */
}
trainAndPredict()
