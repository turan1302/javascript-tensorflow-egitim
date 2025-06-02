/*
🎯 Hedef:
Aynı veri setiyle 3 model eğit

Aynı yapıda ama farklı aktivasyon

Sonuçları karşılaştır: Kayıp (loss) ve tahmin
 */

const tf = require("@tensorflow/tfjs-node")
const {model} = require("@tensorflow/tfjs-node");

const xsRaw = [
    [0, 20, 0, 1],
    [1, 22, 1, 0],
    [2, 25, 0, 0],
    [3, 27, 1, 0],
    [4, 30, 0, 0],
    [5, 28, 1, 1],
    [6, 26, 0, 1],
    [0, 21, 1, 1],
    [2, 23, 1, 0],
    [5, 29, 1, 1]
]

const ysRaw = [
    [100],
    [120],
    [140],
    [180],
    [200],
    [220],
    [190],
    [110],
    [160],
    [210]
]

// normalizasyon fonksiyonu
function normalize(data){
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis noyunca max
    const normalize = dataT.sub(min).div(max.sub(min)) // normalizasyon uyguladık

    return {normalize,min,max}
}

// normalizasyon uygulayalım
const {normalize : xsNorm, min : xsMin, max : xsMax} = normalize(xsRaw)
const {normalize : ysNorm, min : ysMin, max : ysMax} = normalize(ysRaw)

// model oluşturma fonksiyonunu yazalım
function createModel(activation){
    const model = tf.sequential()

    // giris katmanı
    model.add(tf.layers.dense({
        units : 16, // 16 nöron
        inputShape : [4], // 4 giriş
        activation : activation,
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.01
        })
    }))

    // dropout
    model.add(tf.layers.dropout({
        rate : 0.3
    }))

    // gizli katman
    model.add(tf.layers.dense({
        units : 8,
        activation : activation,
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.01
        })
    }))


    // dropout
    model.add(tf.layers.dropout({
        rate : 0.3
    }))

    // çıkış katmanı
    model.add(tf.layers.dense({
        units : 1,
        activation : "linear"
    }))

    // model derleyelim
    model.compile({
        optimizer : "adam",
        loss : "meanSquaredError"
    })

    return model
}


// modeli eğitelim ve tahmin yaptıralım
async function trainAndPredict(activation){
    const fit_model = await createModel(activation)

    const fit_result = await fit_model.fit(xsNorm,ysNorm,{
        epochs : 300,
        verbose : 0
    })

    await fit_model.save(`file://./model-${activation}`)

    const loadedModel = await tf.loadLayersModel(`file://./model-${activation}/model.json`)

    const inputRaw = tf.tensor2d([[5, 27, 1, 1]])
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))

    const predNorm = loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)

    const trust_analysis = await fit_model.evaluate(xsNorm,ysNorm)

    // ------------------
    console.log(`${activation.toUpperCase()} Aktivasyonu`)
    console.log("Son Kayıp: ",fit_result.history.loss.at(-1)) // en son kaybı aldırdık
    console.log("Doğruluk: ",trust_analysis.dataSync()[0])
    pred.print()
}

function run(){
    trainAndPredict("relu")
    trainAndPredict("sigmoid")
    trainAndPredict("tanh")
}
run()
