const tf = require("@tensorflow/tfjs")

// eğitim verilerini girelim
const xs = tf.tensor1d([1,2,3,4])
const ys = tf.tensor1d([4,7,10,13]) // formül: 3x+1

// model oluşturalım
const model = tf.sequential()

model.add(tf.layers.dense({
    units : 1, // 1 nöron kullanılacak. Gelen veriyi bir ağırlık ile çarpıp sapma ekleuecek. 1 değil de başka sayı olsaydı mesela 3 olsaydı o kadar farklı çıktı üretirdi
    inputShape : [1] // bir sayı girilecek. Mesela a3 gibi 5 gibi 50 gibi sallıyorum
}))

// modeli derleyelim
model.compile({
    optimizer : "sgd",  // nöronlar modelin ağırlıklarını vs hesaplıyordu ya. O ağırlık hesaplamada en iyi cevap veren algoritmadır
    loss : "meanSquaredError" // Gerçek değer ile tahmin değer arasındaki farkın karesinin ortalamasını alır. Modelin hatasını buna göre ölçer
})

async function trainAndPredict(){
    await model.fit(xs,ys,{
        epochs : 300 // 300 tekrarla modeli eğitelim
    })

    const sonuc = model.predict(tf.tensor1d([5]))
    sonuc.print()
}

trainAndPredict()
