// NORMALİZASYON: Örnek: Araba Bilgilerine Göre Satış Fiyatı ve Sigorta Ücreti Tahmini
const tf = require("@tensorflow/tfjs-node")

// Girdi verileri: [Yıl, Motor, KM]
const xsRaw = [
    [2015, 1.4, 120000],
    [2018, 1.6, 80000],
    [2020, 2.0, 50000],
    [2022, 1.6, 20000],
    [2023, 1.0, 10000],
]

// Çıkış verileri: [Fiyat, Sigorta]
const ysRaw = [
    [350000, 6000],
    [450000, 5500],
    [650000, 5000],
    [800000, 4500],
    [750000, 4000],
]

// normalizasyon
function normalize(data){
    const dataT = tf.tensor2d(data)
    const max = dataT.max(0)
    const min = dataT.min(0)
    const normalize = dataT.sub(min).div(max.sub(min))
    return {normalize,max,min}
}

// normalize uygulayalım
const {normalize:xsNorm,max:xsMax,min:xsMin} = normalize(xsRaw)
const {normalize:ysNorm,max:ysMax,min:ysMin} = normalize(ysRaw)

// model oluşturalım
const model = tf.sequential()
model.add(tf.layers.dense({
    units : 2,
    inputShape : [3],
    activation : "relu"
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
    const inputRaw = tf.tensor2d([[2021, 1.6, 40000]])
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))
    const predNorm = loadedModel.predict(inputNorm)

    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)

    pred.print()
}

trainAndPredict()
