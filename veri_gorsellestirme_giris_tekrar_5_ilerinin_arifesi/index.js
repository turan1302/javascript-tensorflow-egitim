/*
    ☕ Senaryo: Günlük Kahve Tüketimini Tahmin Et
📋 Amaç:
Bir kişinin günde kaç bardak kahve içeceğini tahmin eden bir makine öğrenimi modeli oluşturacağız.

🔢 Girdiler (Features):
Uyku Süresi (saat)

Toplam Çalışma Süresi (saat)

Stres Seviyesi (0-10 arası)

🎯 Çıktı (Label):
Günlük içilen kahve miktarı (bardak)
 */

const tf = require("@tensorflow/tfjs-node")
const {plot} = require("nodeplotlib");

// giriş verileri
// Her satır: [Günlük içilen kahve miktarı (bardak)]
const xsRaw = [
    [8, 6, 3],
    [5, 10, 8],
    [7, 8, 5],
    [6, 9, 7],
    [4, 12, 9],
    [9, 4, 2],
    [8, 5, 3],
    [6, 11, 8],
    [7, 7, 4],
    [5, 9, 7],
    [6, 10, 6],
    [7, 6, 4],
    [4, 11, 9],
    [9, 5, 2],
    [8, 4, 3],
    [5, 12, 9],
    [6, 8, 5],
    [7, 9, 6],
    [8, 7, 4],
    [4, 13, 10]
]

// çıkış verileri
const ysRaw = [
    [1],
    [4],
    [2],
    [3],
    [5],
    [0],
    [1],
    [4],
    [2],
    [3],
    [3],
    [2],
    [5],
    [0],
    [1],
    [5],
    [2],
    [3],
    [2],
    [6]
]

// normalizasyon fonksiyonu
function normalize(data){
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normalize = dataT.sub(min).div(max.sub(min)) // normalizasyon formülü

    return {normalize,min,max}
}

// normalizasyon uygulayalım
const {normalize : xsNorm, min : xsMin , max : xsMax} = normalize(xsRaw)
const {normalize : ysNorm, min : ysMin , max : ysMax} = normalize(ysRaw)

// model oluşturma
async function createModel(){
    const model = tf.sequential()

    // model ilk katman
    model.add(tf.layers.dense({
        units : 16,
        inputShape : [3],
        activation : "relu",
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.01
        })
    }))

    // gizli katman dropout
    model.add(tf.layers.dropout({
        rate : 0.3
    }))

    // gizli katman model derinleşmesi
    model.add(tf.layers.dense({
        units : 8,
        activation : "relu",
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.01
        })
    }))

    // gizli katman dropout
    model.add(tf.layers.dropout({
        rate : 0.3
    }))

    // gizli katman model derinleşmesi
    model.add(tf.layers.dense({
        units : 4,
        activation : "relu",
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.01
        })
    }))

    // gizli katman dropout
    model.add(tf.layers.dropout({
        rate : 0.3
    }))

    // gizli katman model derinleşmesi
    model.add(tf.layers.dense({
        units : 2,
        activation : "relu",
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.01
        })
    }))

    // gizli katman dropout
    model.add(tf.layers.dropout({
        rate : 0.3
    }))

    // çıkış katmanı
    model.add(tf.layers.dense({
        units : 1,
        activation : "linear",
    }))

    // model derleyelim
    model.compile({
        optimizer : "adam",
        loss : "meanSquaredError"
    })

    return model
}

// model eğitme ve öğrenme
async function trainAndPredict(){
    const fit_model = await createModel()

    const lossValues = []

    const fit_model_result = await fit_model.fit(xsNorm,ysNorm,{
        epochs : 250,
        callbacks : {
            onEpochEnd : async (epoch, logs) => {
                lossValues.push(logs.loss)
            }
        }
    })

    await fit_model.save("file://./model")

    const loadedModel = await tf.loadLayersModel("file://./model/model.json")

    // giriş vs verilerini ayarlayalım
    const inputRaw = tf.tensor2d([[4,16,7]])
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))

    // tahmin ettirelim
    const predNorm = await loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMax)

    // doğruluk analizi
    const trust_analysis = await fit_model.evaluate(xsNorm,ysNorm)

    console.log("Son Model Kaybı: ",fit_model_result.history.loss.at(-1))
    console.log("Doğruluk: ",trust_analysis.dataSync()[0])
    pred.print()

    // plot kısmı
    plot([{
        x : Array.from({length : lossValues.length},(_,i)=>i+1),
        y : lossValues,
        type : "line",
        name : "Loss"
    }],{
        title : "Kayıp Grafiği (Loss)",
        xaxis : {
            title : "Epochs"
        },
        yaxis : {
            title : "Kayıp Değeri"
        }
    })
}

trainAndPredict()
