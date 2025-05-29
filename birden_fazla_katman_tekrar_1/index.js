// birden fazla katman oluşturarak bir örnek daha yapalım

const tf = require("@tensorflow/tfjs-node")

const xs = tf.tensor2d([[1],[2],[3],[4],[5]]) // giriş verileri
const ys = tf.tensor2d([[3],[6],[9],[12],[15]]) // çıkış verileri. y=3x ilişkisi

const model = tf.sequential()

// giriş katmanı
model.add(tf.layers.dense({
    units : 4, // 4 nöron
    inputShape : [1], // 1 giriş
    activation : "relu" // negatifleri temizle dedik
}))

// gizli katman
model.add(tf.layers.dense({
    units : 3, // 3 nöron
    activation : "relu" // negartifleri temizle
}))

// çıkış katmanı
model.add(tf.layers.dense({
    units : 1,  // 1 nöron olacak
    activation : "linear" // sadece çıktı alacağımız için linear olarak belirttik
}))

// modeli compile edelim
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
    const sonuc = loadedModel.predict(tf.tensor2d([[6]]))
    sonuc.print()

    /*
        çıktı:

        [[18]]
     */
}
trainAndPredict()
