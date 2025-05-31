/*
Veri Normalizasyonu / Standardizasyon

Gerçek veriler genellikle çok farklı aralıklarda olur,
bu yüzden giriş ve çıkış verilerini normalize etmek
(0-1 arası, veya standartlaştırmak) modeli çok daha hızlı ve sağlıklı eğitir.
 */

const tf = require("@tensorflow/tfjs-node")

// giriş ve çıkış verileri
const xsRaw = [
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
];

const ysRaw = [
    [10, 11],
    [16, 15],
    [22, 19],
    [28, 23],
    [34, 27],
];


/*
Normalizasyon fonksiyonunu yazalım

Tensor’a çeviriyoruz,

Her sütundaki (özellikteki) minimum ve maksimum değerleri buluyoruz,

Her değerden min’i çıkarıp (x - min),

Sonra max-min farkına bölüyoruz,

Böylece tüm değerler 0-1 aralığına geliyor.
 */

function normalize(data) {
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca minimumu al
    const max = dataT.max(0) // axis boyunca maksimumu al
    const normalize = dataT.sub(min).div(max.sub(min))
    return {normalize,max,min}
}

const {normalize : xsNorm,max : xsMax,min : xsMin} = normalize(xsRaw)
const {normalize : ysNorm,max : ysMax,min : ysMin} = normalize(ysRaw)


// model oluşturalım
const model = tf.sequential()
model.add(tf.layers.dense({
    units : 2,
    inputShape : [3],
    activation : "relu" // doğrusal olmayan daha iyi öğrenme için
}))

 model.compile({
     optimizer : "sgd",
     loss : "meanSquaredError"
 })

async function trainAndPredict(){
    await model.fit(xsNorm,ysNorm,{
        epochs : 500
    })

    await model.save("file://./model")

    const loadedModel = await tf.loadLayersModel("file://./model/model.json")

    const inputRaw = tf.tensor2d([[3,4,5]])
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))

    const predNorm =  loadedModel.predict(inputNorm)

    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)

    pred.print()
}

trainAndPredict()
