// normalizasyon ile ilgili örnek yapalım. Örneğin evin oda sayısı ve metrekaresine göre tahmini fiyat ve kira gelirini
// hesaplayan modeli kodlayalım

const tf = require("@tensorflow/tfjs-node")

const xsRaw = [  // oda, metrekare, bulunduğu kat
    [1, 50, 1],
    [2, 75, 2],
    [3, 100, 3],
    [4, 125, 4],
    [5, 150, 5],
];

const ysRaw = [  // satış fiyatı, kira
    [500000, 2000],
    [750000, 3000],
    [1000000, 4000],
    [1250000, 5000],
    [1500000, 6000],
];


// normalize fonksiyonunu yazalım
function normalize(data) {
    const dataT = tf.tensor2d(data) // tensor oluşturalım
    const min = dataT.min(0) // axis boyunca mminimumu aldık
    const max = dataT.max(0) // axis boyunca maksimumu aldık
    const normalize = dataT.sub(min).div(max.sub(min))

    return {normalize, max, min}
}

// normalize işlemlerinden gelen sonuçlar
const {normalize: xsNorm, max: xsMax, min: xsMin} = normalize(xsRaw)
const {normalize: ysNorm, max: ysMax, min: ysMin} = normalize(ysRaw)

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

    const inputRaw = tf.tensor2d([[3, 110, 2]])
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))

    const predNorm = loadedModel.predict(inputNorm)

    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)
    pred.print()
}

trainAndPredict()

