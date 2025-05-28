const tf = require("@tensorflow/tfjs-node")

// eğitim verileri
const xs = tf.tensor1d([2,4,6,8])
const ys = tf.tensor1d([6,10,14,18]) // y = 2x+2

const model = tf.sequential()
model.add(tf.layers.dense({
    units : 1,
    inputShape : [1]
}))

model.compile({
    optimizer : "sgd",
    loss : "meanSquaredError"
})

async function trainAndPredict(){
    await model.fit(xs,ys,{
        epochs : 200
    })

    await model.save("file://./model")

    console.log("Model Kaydedildi")

    const loadedModel = await tf.loadLayersModel("file://./model/model.json")
    const sonuc = loadedModel.predict(tf.tensor1d([10]))
    sonuc.print()
}

trainAndPredict();
