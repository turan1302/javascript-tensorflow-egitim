// birden fazla katman kullanımı tekrar edelim

const tf = require("@tensorflow/tfjs-node")

const xs = tf.tensor2d([[1],[2],[3],[4],[5]])
const ys = tf.tensor2d([[7],[9],[11],[13],[15]]) // 2x+5 formülü

const model = tf.sequential()

// giriş katmanı
model.add(tf.layers.dense({
    units : 3, // 3 nöron
    activation : "relu",  // negatifleri düzenle dedik
    inputShape : [1]  // 1 veri girecek
}))

// gizli katman
model.add(tf.layers.dense({
    units : 2,
    activation : "relu"
}))

// çıkış katmanı
model.add(tf.layers.dense({
    units : 1,
    activation : "linear"
}))


// compile edelim yani modelimizi derleyelim
model.compile({
    optimizer  :"sgd",
    loss : "meanSquaredError"
})

async function trainAndPredict(){
    await model.fit(xs,ys,{
        epochs : 200
    })

    await model.save("file://./model")

    const loadedModel = await tf.loadLayersModel("file://./model/model.json")
    const sonuc = loadedModel.predict(tf.tensor2d([[6]]))
    sonuc.print();

    // doğruluk testini de burada yapalım
    const result = await model.evaluate(tf.tensor2d([[6]]),tf.tensor2d([[17]]))
    console.log("Doğruluk Analizi: ",result.dataSync()[0])

}

trainAndPredict()
