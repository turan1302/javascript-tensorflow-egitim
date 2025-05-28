// örnek olarak makinamızı eğitelim

const tf = require("@tensorflow/tfjs")

// eğitim için giriş ve çıktı verilerini verelim
const xs = tf.tensor1d([2,4,6,8])
const ys = tf.tensor1d([5,9,13,17])  // formil = 2x+1 oluyor

const model = tf.sequential()
model.add(tf.layers.dense({
    units : 1, // 1 nöron kullanılacak. Gelen veriyi bir ağırlık ile çarpıp sapma ekleuecek. 1 değil de başka sayı olsaydı mesela 3 olsaydı o kadar farklı çıktı üretirdi
    inputShape : [1]  // bir sayı girilecek. Mesela a3 gibi 5 gibi 50 gibi sallıyorum
}))

model.compile({
    optimizer  : "sgd",
    loss : "meanSquaredError"
})


async function trainAndPredict(){
    await model.fit(xs,ys,{
        epochs : 200        // giriş ve çıkış verilerini al, modeli 500 kez eğit
    })

    const sonuc = model.predict(tf.tensor1d([10]))
    sonuc.print()
}

trainAndPredict()
