const tf = require("@tensorflow/tfjs-node")

const xs = tf.tensor1d([1,3,5,7])
const ys  = tf.tensor1d([4,10,16,22]) // y = 3x+1

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
        epochs : 400
    })

    await model.save("file://./model3")

    const loadedModel = await tf.loadLayersModel("file://./model3/model.json")
    const sonuc = loadedModel.predict(tf.tensor1d([9]))
    sonuc.print()

    /*
        sonuc:

Tensor
     [[28.0465126],]
     */
}
trainAndPredict()
