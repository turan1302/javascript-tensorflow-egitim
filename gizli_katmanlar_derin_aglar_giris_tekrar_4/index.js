/*
derin ağ örneği son tekrarımızı yapalım

💡 Amaç:
Derin katmanlar

Regularization

Dropout

Tahmin çıktısı

Aktivasyonların farkı için zemin hazırlamak
 */

const tf = require("@tensorflow/tfjs-node")

// Girdi: [gün, sıcaklık, indirim var mı, hafta sonu mu]
const xsRaw = [
    [0, 20, 0, 1],
    [1, 22, 1, 0],
    [2, 25, 0, 0],
    [3, 27, 1, 0],
    [4, 30, 0, 0],
    [5, 28, 1, 1],
    [6, 26, 0, 1],
    [0, 21, 1, 1],
    [2, 23, 1, 0],
    [5, 29, 1, 1]
]

// Çıktı: [sipariş sayısı]
const ysRaw = [
    [100],
    [120],
    [140],
    [180],
    [200],
    [220],
    [190],
    [110],
    [160],
    [210]
]

// normalizasyon fonksiyonu yazalım
function normalize(data){
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normalize = dataT.sub(min).div(max.sub(min)) // normalizasyon fonksiyonu
    return {normalize,min,max}
}

// normalize uygulayalım
const {normalize : xsNorm, min : xsMin, max : xsMax} = normalize(xsRaw)
const {normalize : ysNorm, min : ysMin, max : ysMax} = normalize(ysRaw)

// model oluşturalım
const model = tf.sequential()

// model giriş katmanı
model.add(tf.layers.dense({
    units : 32, // 32 nöron
    inputShape : [4], // 4 girdi
    activation : "relu", // karmaşık öğrenme modeli için yazdık
    kernelRegularizer : tf.regularizers.l2({  // hata katsayısı için ceza uygulayalım. Overfitting yapmasın
        l2 : 0.01
    })
}))

// overfit olmaması için yüzde 30 nöronu silelim ()
model.add(tf.layers.dropout({
    rate : 0.3
}))

// gizli katman yapalım
model.add(tf.layers.dense({
    units : 16, // 16 nöron
    activation : "relu",  // karmaşık öğrenme modeli için yazdık
    kernelRegularizer : tf.regularizers.l2({  // hata katsayısı için ceza uygulayalım. Overfitting yapmasın
        l2 : 0.01
    })
}))

// overfit olmaması için yüzde 30 nöronu silelim ()
model.add(tf.layers.dropout({
    rate : 0.3
}))

// gizli katman yapalım
model.add(tf.layers.dense({
    units : 8, // 16 nöron
    activation : "relu",  // karmaşık öğrenme modeli için yazdık
    kernelRegularizer : tf.regularizers.l2({  // hata katsayısı için ceza uygulayalım. Overfitting yapmasın
        l2 : 0.01
    })
}))

// overfit olmaması için yüzde 30 nöronu silelim ()
model.add(tf.layers.dropout({
    rate : 0.3
}))

// çıkış katmanı yapalım
model.add(tf.layers.dense({
    units : 1, // 1 nöron
    activation : "linear" // direkt çıktıyı ver dedil. Ekstra bir şey yapmasın
}))

// modeli derleyelim
model.compile({
    optimizer : "adam",
    loss : "meanSquaredError"
})

// modeli eğitelim
async function trainAndPredict(){
    const fit_result = await model.fit(xsNorm,ysNorm,{
        epochs : 500
    })

    // model öğrenme kaybı
    console.log("Öğrenme Kaybı: ",fit_result.history.loss)


    await model.save("file://./model")

    const loadedModel = await tf.loadLayersModel("file://./model/model.json")

    const inputRaw = tf.tensor2d([[5, 27, 1, 1]])
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))

    const predNorm = await loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)

    pred.print()

    // doğruluk analizi
    const trust_analys = await model.evaluate(xsNorm,ysNorm)
    console.log("Doğruluk Analizi: ",trust_analys.dataSync()[0])
}
trainAndPredict()
