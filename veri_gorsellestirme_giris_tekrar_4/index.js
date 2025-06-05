/*
🎯 Senaryo: Bir kişinin günlük su içme miktarını tahmin eden bir model yapacağız.
🔢 Girdiler (Features):
Günlük adım sayısı

Hava sıcaklığı (°C)

Egzersiz süresi (dakika)

🎯 Çıktı (Label):
Günlük içilen su miktarı (litre)
 */

const tf = require("@tensorflow/tfjs-node")
const {plot} = require("nodeplotlib");

// giris verileri
const xsRaw = [
    [4000, 20, 0],    // 4000 adım, 20 derece, 0 dk egzersiz
    [8000, 25, 30],
    [3000, 18, 0],
    [10000, 30, 60],
    [6000, 22, 15],
    [12000, 32, 90],
    [2000, 17, 0],
    [15000, 35, 120],
    [5000, 21, 10],
    [7000, 24, 20]
]

// cikis verileri
const ysRaw = [
    [1.2],   // litre
    [2.0],
    [1.0],
    [2.5],
    [1.5],
    [3.0],
    [0.9],
    [3.5],
    [1.3],
    [1.7]
]

// normalizasyon fonksiyonu
function normalize(data){
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normalize = dataT.sub(min).div(max.sub(min)) // normalize formülü
    return {normalize, min, max}
}

// normalizasyon uygulayalım
const {normalize : xsNorm, min : xsMin, max : xsMax} = normalize(xsRaw)
const {normalize : ysNorm, min : ysMin, max : ysMax} = normalize(ysRaw)

// model oluşturma fonksiyonumuz
async function createModel(){
    const model = tf.sequential()

    // katman 1
    model.add(tf.layers.dense({
        units : 16,
        inputShape : [3],
        activation : "relu",
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.01
        })
    }))

    model.add(tf.layers.dropout({
        rate : 0.3
    }))

    model.add(tf.layers.dense({
        units : 8,
        activation : "relu",
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.01
        })
    }))

    model.add(tf.layers.dropout({
        rate : 0.3
    }))


    model.add(tf.layers.dense({
        units : 4,
        activation : "relu",
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.01
        })
    }))

    model.add(tf.layers.dropout({
        rate : 0.3
    }))

    model.add(tf.layers.dense({
        units : 2,
        activation : "relu",
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.01
        })
    }))

    model.add(tf.layers.dropout({
        rate : 0.3
    }))

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

// model eğit ve tahmin yap
async function trainAndPredict(){
    const fit_model = await createModel()

    const lossValues = []

    const fit_model_result = await fit_model.fit(xsNorm,ysNorm,{
        epochs : 250,
        callbacks : {
            onEpochEnd : (epoch, logs) => {
                lossValues.push(logs.loss)
            }
        }
    })

    await fit_model.save("file://./model")

    const loadedModel = await tf.loadLayersModel("file://./model/model.json")

    const inputRaw = tf.tensor2d([[9000, 28, 45]])
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))

    const predNorm = await loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)

    // doğruluk analizi
    const trust_analysis = await fit_model.evaluate(xsNorm, ysNorm)

    console.log("Son Eğitim Kaybı: ",fit_model_result.history.loss.at(-1))
    console.log("Doğruluk: ",trust_analysis.dataSync()[0])


    pred.print()

    plot([{
        x : Array.from({length : lossValues.length},(_,i)=>i+1),
        y : lossValues,
        type : "line",
        name : "Loss"
    }],{
        title : "Kayıp Değerleri (Loss)",
        yaxis : {
            title : "Kayıp Değerleri"
        },
        xaxis : {
            title : "Epochs"
        }
    })
}

trainAndPredict()
