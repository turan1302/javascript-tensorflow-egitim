// coklu veri girisi/cıkısı olayları için örnek yapalım

const tf = require("@tensorflow/tfjs-node")

const xs = tf.tensor2d([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7]
])

const ys = tf.tensor2d([
    [20, 5],
    [29, 5],
    [38, 9],
    [47, 11],
    [56, 13]
])

// model oluşturalım
const model = tf.sequential()

// model katmanı oluşturalım
model.add(tf.layers.dense({
    units : 2, // 2 çıkış
    inputShape : [3], // 3 giriş
    activation : "linear" // sadece çıktı verecek ekstra bir hesaplama yapmayacak
}))

// model derleyelim
model.compile({
    optimizer : "sgd",
    loss : "meanSquaredError"
})

async function trainAndPredict(){
    await model.fit(xs,ys,{
        epochs : 300
    })

    await model.save("file://./model")

    const loadedModel = await tf.loadLayersModel("file://./model/model.json")
    const sonuc = loadedModel.predict(tf.tensor2d([[6,7,8]]))
    sonuc.print()

    const r_squared = await model.evaluate(tf.tensor2d([[6,7,8]]),tf.tensor2d([[65,15]]))
    console.log("Doğruluk: ",r_squared.dataSync()[0])
}
trainAndPredict()

