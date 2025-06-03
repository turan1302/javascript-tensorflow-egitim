/*
📊 Yeni Örnek Fikri: Öğrenci Not Tahmini
🎯 Amaç:
Bir öğrencinin:

kaç saat ders çalıştığı,

derse katılım durumu (1: var, 0: yok),

uyku süresi,

ödev teslim durumu (1: yaptı, 0: yapmadı)

bilgilerine göre tahmini notunu (0-100 arası) bulalım.

Ve yine relu, sigmoid, tanh aktivasyonlarını karşılaştıralım.

 */

const xsRaw = [
    [2, 1, 6, 1],  // az çalıştı ama dersi dinledi ve ödev yaptı
    [4, 0, 7, 0],  // çalıştı ama ilgisizdi
    [1, 0, 5, 0],  // çalışmadı ilgisizdi
    [5, 1, 6, 1],  // iyi çalıştı, derse katıldı
    [3, 1, 7, 1],
    [6, 1, 6, 1],
    [2, 0, 6, 0],
    [7, 1, 8, 1],
    [4, 1, 7, 1],
    [1, 0, 5, 0]
]

const ysRaw = [
    [70],
    [60],
    [40],
    [90],
    [80],
    [95],
    [50],
    [100],
    [85],
    [35]
]

const tf = require("@tensorflow/tfjs-node")
const {acos} = require("@tensorflow/tfjs-node");

// normalizasyon fonksiyonu yazalım
function normalizee(data){
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normalize = dataT.sub(min).div(max.sub(min)) // normalize uyguladık

    return {normalize,min,max}
}

// normalize uygulayalım
const {normalize : xsNorm, min : xsMin, max : xsMax} = normalizee(xsRaw)
const {normalize : ysNorm, min : ysMin, max : ysMax} = normalizee(ysRaw)

// model olşturan fonksiyon yazalım
async function createModel(activation){
    const model = await tf.sequential()

    model.add(tf.layers.dense({
        units : 16, // 16 nöron
        inputShape : [4], // 4 veri girecek
        activation : activation,
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.01
        })
    }))

    // drop koyalım eğitimde ezberlememesi için
    model.add(tf.layers.dropout({
        rate : 0.3
    }))

    model.add(tf.layers.dense({
        units : 8, // 8 nöron
        activation : activation,
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.01
        })
    }))

    // drop koyalım eğitimde ezberlememesi için
    model.add(tf.layers.dropout({
        rate : 0.3
    }))

    // çıkış katmanı
    model.add(tf.layers.dense({
        units : 1, // 1 nöron
        activation : "linear",
    }))

    // model derleyelim
    model.compile({
        optimizer : "adam",
        loss : "meanSquaredError"
    })

    return model
}


async function trainAndPredict(activation){
    // modeli çağıralım
    const model = await createModel(activation)

    // modeli eğitelim
    const fit_model = await model.fit(xsNorm,ysNorm,{
        epochs : 500
    })

    // modelşi kaydedelm
    await model.save(`file://./model-${activation}`)

    const loadedModel = await tf.loadLayersModel(`file://./model-${activation}/model.json`)

    // veri giriş
    const inputRaw = tf.tensor2d([[3, 1, 7, 1]])
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))

    // tahmin
    const predNorm = await loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)

    // doğruluk
    const trust_analysis = await model.evaluate(xsNorm,ysNorm)

    console.log(`${activation.toUpperCase()} Algoritması`)
    console.log("Son Öğrenme Kaybı: ",fit_model.history.loss.at(-1))
    console.log("Doğruluk: ",trust_analysis.dataSync()[0])
    pred.print()
    console.log("--------------------------------------------------------------")
}

function run(){
    trainAndPredict("relu")
    trainAndPredict("sigmoid")
    trainAndPredict("tanh")
}
run()
