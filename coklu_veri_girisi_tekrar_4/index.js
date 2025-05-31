// coklu veri girisi / cıkısı tekrar

const tf = require("@tensorflow/tfjs-node")

const xs = tf.tensor2d([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7]
])

const ys = tf.tensor2d([
    [10, 11],
    [16, 15],
    [22, 19],
    [28, 23],
    [34, 27]
])


// model oluşturalım
const model = tf.sequential()

// model katmanı
model.add(tf.layers.dense({
    units: 2, // çıktı
    inputShape: [3], // girdi
    activation: "linear"
}))

// model derleyelim
model.compile({
    optimizer : "sgd",
    loss : "meanSquaredError"
})

async function trainAndPredict(){
    await model.fit(xs,ys,{
        epochs : 500
    })

    await model.save("file://./model")

    const loadedModel = await tf.loadLayersModel("file://./model/model.json")
    const sonuc = loadedModel.predict(tf.tensor2d([[9,10,11]]))
    sonuc.print()

    const r_squared = await model.evaluate(tf.tensor2d([[9,10,11]]),tf.tensor2d([[58,43]]))
    console.log("Doğruluk Analizi: ",r_squared.dataSync()[0])
}

trainAndPredict()
