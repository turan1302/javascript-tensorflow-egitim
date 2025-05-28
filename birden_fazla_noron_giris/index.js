// birden fazla nöronlarla çalışalım.

// bu işlem en basitinden günlük hayatta hava durumu projelerinde kullanılabilir.

const tf = require("@tensorflow/tfjs-node")

const xs = tf.tensor2d([[1],[2],[3],[4]])  // giriş verisi
const ys = tf.tensor2d([[2,4],[4,8],[6,12],[8,16]]) // çıkış verisi

// bir bu kısımda şimdi 3 girince mesela karşımıza [6,12] çıkmasını isteyeceğiz
// bunun için de 2 sonuç alacağımız içim NÖRON 2 olacak. Yani 2 farklı çıktı üretecek

const model = tf.sequential()
model.add(tf.layers.dense({
    units : 2,  // 2 farklı çıktı gelecek
    inputShape : [1]  // tek giriş olacak
}))

// modeli derleyelim
model.compile({
    optimizer : "sgd",
    loss: 'meanSquaredError'
})

async function trainAndPredict(){
    await model.fit(xs,ys,{
        epochs : 500
    })

    await model.save("file://./model_multi_output")

    const loadedModel = await tf.loadLayersModel("file://./model_multi_output/model.json")

    const sonuc = loadedModel.predict(tf.tensor2d([[5]]))  // [10,20] olarak çıktı gelecek tahminen
    sonuc.print()
}
trainAndPredict()


