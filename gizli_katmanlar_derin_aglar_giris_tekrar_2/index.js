/*
🎯 Yeni Konu: Overfitting ile Mücadele — Dropout ve Regularization
Bu örnekte, amacımız modelin aşırı öğrenmesini (overfitting) engellemek olacak. Modeli çok iyi eğitirsek, eğitim verisinde süper sonuç alabilir ama yeni verilere karşı kötü sonuçlar verir. İşte bunu önlemek için:

📘 Konu Başlığı:
Dropout ve L2 Regularization Kullanarak Overfitting'i Azaltma

🎯 Görev:
Girdi: [yaş, günlük kalori alımı, egzersiz süresi (dakika), uyku süresi (saat)]

Çıktı: [vücut yağı yüzdesi]


🔧 Yapılacaklar:
Normalizasyon (önceki fonksiyonu kullanabilirsin)

Model:

dense(16, activation: 'relu', kernelRegularizer: l2)

dropout(0.3)

dense(8, activation: 'relu')

dense(1, activation: 'linear')

Optimizer: adam

Loss: meanSquaredError

Epoch: 600

Eğitim sonrası [25, 2500, 45, 7] girdisiyle vücut yağı tahmini al
 */

const tf = require("@tensorflow/tfjs-node")

const xsRaw = [
    [20, 2200, 30, 6],
    [25, 2500, 45, 7],
    [30, 2800, 60, 8],
    [35, 2000, 20, 5],
    [40, 1800, 15, 5],
    [28, 2700, 50, 7],
    [22, 2300, 35, 6],
    [38, 1900, 25, 6]
]

const ysRaw = [
    [18],
    [16],
    [15],
    [22],
    [25],
    [17],
    [19],
    [23]
]

// normalizasyon fonksiyonu oluşturalım
function normalize(data){
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // aixs boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normalize = dataT.sub(min).div(max.sub(min))
    return {normalize,min,max}
}

// normalize uygulayalım
const {normalize : xsNorm,min : xsMin,max : xsMax} = normalize(xsRaw)
const {normalize : ysNorm,min : ysMin,max : ysMax} = normalize(ysRaw)

// model oluşturalım
const model = tf.sequential()

// giriş katmanı
model.add(tf.layers.dense({
    units : 16, // 16 nöron,
    inputShape : [4], // 4 giriş
    activation : "relu", // karmaşık veri analizi olacak. 0 ları görmezden gel
    kernelRegularizer : "l1l2" // overfitting durumlarında katman ağırlıklarına ceza katsayısı uygula. Amaç modelin karmaşık olmaması, genelleme yapabilmesi
}))

// gizli katman
model.add(tf.layers.dropout({
    rate : 0.3  // karmaşık öğrenme olmasın diye nöronların yüzde 30 unu kapattık
}))

// gizli katman
model.add(tf.layers.dense({
    units : 8, // 8 nöron,
    activation : "relu", // karmaşık veri analizi olacak. 0 ları görmezden gel
}))

// çıkış katmanı
model.add(tf.layers.dense({
    units : 1, // 1 nöron,
    activation : "linear", // direkt çıktıyı basmak istedik
}))

// model derleyelim
model.compile({
    optimizer : "adam",
    loss : "meanSquaredError"
})

// model eğitelim ve tahmin yaptıralım
async function trainAndPredict(){
    const fit_result = await model.fit(xsNorm,ysNorm,{
        epochs : 600
    })

    await model.save("file://./model")

    const loadedModel = await tf.loadLayersModel("file://./model/model.json")
    const inputRaw = tf.tensor2d([[24, 1300, 5, 9]])
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))

    const predNorm = model.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)

    pred.print()

    console.log("Hata Değeri: ",fit_result.history.loss)

    const trust_value = await model.evaluate(xsNorm,ysNorm)

    console.log("Doğruluk Yüzdesi: ",trust_value.dataSync()[0])

    /*
    tahmini beklenen çıktı: 15.5 - 16.5 arası
     */
}
trainAndPredict()
