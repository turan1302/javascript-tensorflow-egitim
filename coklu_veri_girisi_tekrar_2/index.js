// burada ise artık 2 giriş, 2 çıkış yapalım.
// gerçek hayattaki veri örneklerinde de olacak veri işlemleri için hazırlık denebilir :)

const tf = require("@tensorflow/tfjs-node")

const xs = tf.tensor2d([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
])

const ys = tf.tensor2d([
    [8,-1],
    [13,-1],
    [18,-1],
    [23,-1]
])


// model oluşturalım
const model = tf.sequential()

// model katmanı
model.add(tf.layers.dense({
    units : 2, // 2 çıktı
    inputShape : [2], // 2 veri girişi
    activation : "linear"
}))

// model derleme
model.compile({
    optimizer : "sgd",
    loss : "meanSquaredError"
})


async function trainAndPredict(){
    // model eğitimi
    await model.fit(xs,ys,{
        epochs : 400
    })

    await model.save("file://./model")

    const loadedModel = await tf.loadLayersModel("file://./model/model.json")
    const sonuc = loadedModel.predict(tf.tensor2d([[5,6]]))
    sonuc.print()

    const r_squared = await model.evaluate(tf.tensor2d([[7,8]]),tf.tensor2d([[38,-1]]))
    console.log("Doğruluk: ",r_squared.dataSync()[0])
}

trainAndPredict()
