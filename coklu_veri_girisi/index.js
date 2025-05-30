// coklu veri girisi konusuna girelim
// buradaki amac bizim modeli eğitirken [[3]] değil de [[3,6]] şeklinde veri girmemiz olatyıdır

// ileriki zamanlarda birden fazla veri giriş olayları ile tek bir sonuç , çok fazla bir sonuç alma işlemlerine de girebiliriz
// bu konu onun için faydalıdır

const tf = require("@tensorflow/tfjs-node")
const {tensor2d} = require("@tensorflow/tfjs-node");

const xs = tf.tensor2d([
    [1,2],
    [2,3],
    [3,4],
    [4,5]
])

const ys = tf.tensor2d([[8],[13],[18],[23]])  // formül: 2 * x1 + 3 * x2 = y


// model oluştualım
const model = tf.sequential()

// model katmanı
model.add(tf.layers.dense({
    units : 1, // 1 çıktı
    inputShape : [2], // 2 giriş
    activation : "linear" // direkt sonucu bekliyoruz
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
    const sonuc = loadedModel.predict(tf.tensor2d([[5,6]]))
    sonuc.print()
    const r_squared = await model.evaluate(tf.tensor2d([[6,7]]),tf.tensor2d([[33]]))
    console.log("Doğruluk: ",r_squared.dataSync()[0])
}

trainAndPredict()
