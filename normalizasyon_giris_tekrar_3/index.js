/*
 Örnek Konu: Kargonun Teslim Süresi ve Ücret Tahmini
🎯 Amaç:
Bir kargonun;

Gönderildiği şehir ile varış şehri arası mesafeye (km),

Kargo ağırlığına (kg),

Gönderim tipine (1: Standart, 2: Hızlı, 3: Ekspres)

göre teslim süresi (gün) ve kargo ücreti (₺) tahmini yapacağız.
 */

const tf = require("@tensorflow/tfjs-node")

const xsRaw = [
    [100, 1, 1],   // kısa mesafe, hafif, standart
    [250, 5, 2],   // orta mesafe, orta ağırlık, hızlı
    [500, 10, 3],  // uzun mesafe, ağır, ekspres
    [750, 7, 2],   // çok uzun mesafe, orta ağırlık, hızlı
    [50, 2, 1],    // çok kısa mesafe, hafif, standart
];

const ysRaw = [
    [3, 30],    // teslim süresi (gün), ücret
    [2, 50],
    [1, 100],
    [1, 80],
    [4, 25],
];

// normalize fonksiyonunu oluşturalım
function normalize(data){
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normalize = dataT.sub(min).div(max.sub(min)) // normalizasyon fonksiyonu
    return {normalize,min,max}
}

// normalizasyon uygulayalım
const {normalize : xsNorm,min : xsMin,max : xsMax} =  normalize(xsRaw)
const {normalize : ysNorm,min : ysMin,max : ysMax} =  normalize(ysRaw)

// model oluşturalım
const model = tf.sequential()

// model katmanı
model.add(tf.layers.dense({
    units : 2, // 2 çıkış
    inputShape : [3], // 3 giriş
    activation : "relu" // karmaşık işlemler olacak hesaplama yap, aynı zamanda negatifleri 0 yap pozitiflere dokunma dedik
}))

// modeli derle
model.compile({
    optimizer : "sgd",
    loss : "meanSquaredError"
})

// modeli eğit ve tahmin yap
async function trainAndPredict(){
    await model.fit(xsNorm,ysNorm,{
        epochs : 300
    })

    await model.save("file://./model")

    const loadedModel = await tf.loadLayersModel("file://./model/model.json")

    const inputRaw = tf.tensor2d([[300, 4, 2]])
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))

    const predNorm = loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)

    pred.print()

    /*
    tahmini çıktı:
        Modelin tahmini olarak şuna benzer bir sonuç vermesi beklenir:

Teslim süresi: ~2 gün

Kargo ücreti: ~55-65 ₺ civarında
     */
}
trainAndPredict()

