// birden fazla katmanla çalışalım
// bu birden fazla katmanla çalışmak derin öğrenme durumlarında kullanılır
// dünyadaki ilişkiler genelde karmaşıktır

const tf = require("@tensorflow/tfjs-node")

const xs = tf.tensor2d([[1],[2],[3],[4]]) // giriş verileri
const ys = tf.tensor2d([[4],[7],[10],[13]]) // çıkış verileri

const model = tf.sequential()

// giriş katmanı
model.add(tf.layers.dense({
    units : 1,  // 1 nöron
    inputShape : [1], // 1 giriş
    activation : "relu"  // karmaşık yapıları öğrenmesi için yaptık. Eğer pozitifse olduğu gibi bırak, negatifse sıfır yap dedik
}))

// gizli katman
model.add(tf.layers.dense({
    units : 4,
    activation : "relu"
}))

// çıkış katmanı
model.add(tf.layers.dense({
    units : 1,
    activation : "linear" // burada y = x dedik. Yani gelen veriyi hiç değiştirme diyoruz. Sadwce trahmin yapcaz burada
}))

model.compile({
    optimizer : "sgd",  // sgd haricinde tf.train.adam() kullanılabilr
    loss : "meanSquaredError"
})

async function trainAndPredict(){
    await model.fit(xs,ys,{
        epochs : 500
    })

    const sonuc = model.predict(tf.tensor2d([[5]]))
    sonuc.print()

    /*
        çıktı: 16
     */
}
trainAndPredict()
