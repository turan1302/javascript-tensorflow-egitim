// çoklu çıktı kısmı için tekrardan bir örnek yapalım

/*
    giriş verileri : [[2],[4],[6],[8]]
    çıkış verileri: [[5, 8], [9, 14], [13, 20], [17, 26]]

    formül:
        Birinci değer: giriş * 2 + 1
        İkinci değer: giriş * 3 + 2
 */

const tf = require("@tensorflow/tfjs-node")

const xs = tf.tensor2d([[2],[4],[6],[8]])
const ys = tf.tensor2d([[5, 8], [9, 14], [13, 20], [17, 26]])

const model = tf.sequential()
model.add(tf.layers.dense({
    units : 2, // 2 çıktı üretecek
    inputShape : [1] // 1 giriş olacak
}))

model.compile({
    optimizer : "sgd",
    loss : "meanSquaredError"
})

async function trainAndPredict(){
    await model.fit(xs,ys,{
        epochs : 300
    })

    await model.save("file://./model_multi_output")

    const loadedModel = await tf.loadLayersModel("file://./model_multi_output/model.json")

    const sonuc = loadedModel.predict(tf.tensor2d([[10]]))
    sonuc.print()
}
trainAndPredict()
