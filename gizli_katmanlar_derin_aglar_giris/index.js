/*
 Yeni Konu: Gizli Katmanlar ve Daha Derin Ağlar
Bu örnekte, konut fiyatı tahmini için önceki gibi bir veri seti kullanacağız, ama bu sefer:

Daha fazla veri noktası

2 gizli katman

relu ve sigmoid gibi farklı aktivasyon fonksiyonları kullanacağız

-------------------------------------------------------------------------------------------------
 */



/*
Görev: Gizli Katmanlı Model ile Konut Fiyatı ve Kira Tahmini

Girdi Verileri: [oda sayısı, metrekare, bulunduğu kat, bina yaşı]

Çıkış Verileri: [satış fiyatı ₺, kira ₺]

🔧 Yapılacaklar:
Normalizasyon (aynı fonksiyonları kullanabilirsin)

Modeli 2 gizli katmanla oluştur:

katman: dense, 8 nöron, relu

katman: dense, 4 nöron, relu

Çıkış katmanı: 2 nöron, linear

adam optimizatörünü kullan

Epoch: 500

Eğitimden sonra [3, 105, 2, 4] girdisiyle tahmin al
 */

const tf = require("@tensorflow/tfjs-node")

const xsRaw = [  // [oda sayısı, metrekare, bulunduğu kat, bina yaşı]
    [1, 50, 1, 15],
    [2, 75, 2, 10],
    [3, 100, 3, 5],
    [4, 125, 4, 3],
    [5, 150, 5, 1],
    [3, 110, 2, 8],
    [2, 85, 1, 12],
    [4, 130, 3, 6],
];

const ysRaw = [  // [satış fiyatı ₺, kira ₺]
    [500000, 2000],
    [750000, 3000],
    [1000000, 4000],
    [1250000, 5000],
    [1500000, 6000],
    [1100000, 4200],
    [850000, 3200],
    [1300000, 5200],
];

// normalize fonksiyonunu yazalım
function normalize(data){
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min değeri aldık
    const max = dataT.max(0) // axis boyunca max değeri aldık
    const normalize = dataT.sub(min).div(max.sub(min)) // normalize fonksiyonunu ayarladık
    return {normalize,min,max}
}

// normalizasyon uygulayalım
const {normalize : xsNorm,min : xsMin, max : xsMax} = normalize(xsRaw)
const {normalize : ysNorm,min : ysMin, max : ysMax} = normalize(ysRaw)

// model oluşturalım
const model = tf.sequential()

// 1. katman oluşturalım
model.add(tf.layers.dense({
    units : 8, // 8 nöron
    inputShape : [4], // 4 giriş
    activation : "relu" // relu ile karmaşık veriyi analiz et
}))

// 2. katman oluşturalım
model.add(tf.layers.dense({
    units : 4, // 4 nöron
    activation : "relu" // relu ile karmaşık veriyi analiz et
}))

// çıkış katmanı
model.add(tf.layers.dense({
    units : 2, // 2 çıkış
    activation : "linear" // herhangi bir hesaplama veya negatif ayrımına girme (negatifse 0 yapma olan değer sonu. neyse onu yazdır)
}))

// model compile et
model.compile({
    optimizer : "adam",
    loss : "meanSquaredError"
})

// eğitim ve tahmin fonksiyonu
async function trainAndPredict(){
    const fit_result = await model.fit(xsNorm,ysNorm,{
        epochs : 500
    })

    await model.save("file://./model")

    const loadedModel = await tf.loadLayersModel("file://./model/model.json")
    const inputRaw = tf.tensor2d([[3, 105, 2, 4]])
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))

    const predNorm = loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)

    pred.print()

    console.log("Hata Değeri: ",fit_result.history.loss)  // Model öğrenip öğrenmediğini kontrol ettik

    // doğruluk çözümü
    const eval_result = await model.evaluate(xsNorm,ysNorm)
    eval_result.print()

    /*
       tahmin sonucu kabaca şöyle bir şey olacaktır:

Satış Fiyatı: 1.030.000 – 1.080.000 ₺ arası

Kira: 4050 – 4150 ₺ arası
     */

}
trainAndPredict()
