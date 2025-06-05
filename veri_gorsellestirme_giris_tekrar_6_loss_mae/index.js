/*
🧪 Yeni ML Örneği: Sıcaklık + Nem + Rüzgar → Günlük Su Tüketimi (litre)
📋 Senaryo:
Kişinin günlük ortalama içtiği su miktarını tahmin etmek istiyoruz. Etkileyen faktörler:

Sıcaklık (°C)

Nem (%)

Rüzgar Hızı (km/h)
 */

const tf = require("@tensorflow/tfjs-node")
const {plot} = require("nodeplotlib")

// girdi verileri
const xsRaw = [
    [30, 40, 5],
    [35, 30, 10],
    [28, 50, 4],
    [32, 35, 7],
    [40, 20, 15],
    [25, 60, 3],
    [33, 45, 6],
    [38, 25, 12],
    [29, 55, 4],
    [36, 28, 11],
    [27, 65, 3],
    [31, 38, 6],
    [34, 33, 8],
    [26, 70, 2],
    [39, 22, 13],
    [24, 68, 2],
    [37, 26, 12],
    [30, 42, 5],
    [33, 40, 7],
    [28, 58, 4]
]

// çıktı verileri
const ysRaw = [
    [2.5],
    [3.2],
    [2.2],
    [2.8],
    [4.0],
    [1.8],
    [2.9],
    [3.8],
    [2.3],
    [3.4],
    [1.7],
    [2.6],
    [3.0],
    [1.6],
    [3.9],
    [1.5],
    [3.6],
    [2.5],
    [2.9],
    [2.1]
]

// normalizasyon fonksiyonları
function normalize(data) {
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normalize = dataT.sub(min).div(max.sub(min)) // normalizasyon formülü
    return {normalize, min, max}
}

// normalizasyon işlemi
const {normalize: xsNorm, min: xsMin, max: xsMax} = normalize(xsRaw)
const {normalize: ysNorm, min: ysMin, max: ysMax} = normalize(ysRaw)

// model oluşturma fonksiyonu
async function createModel() {
    const model = await tf.sequential()

    // giriş katmanı
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

    // gizli katman
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

    // gizli katman
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

    // gizli katman
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

    // çıkış katman
    model.add(tf.layers.dense({
        units : 1,
        activation : "linear",
    }))

    // model derleyelim
    model.compile({
        optimizer : "adam",
        loss : "meanSquaredError",
        metrics : ["mae"]
    })

    return model
}

// model eğitimi ve tahmin etme
async function trainAndPredict(){
    const fit_model = await createModel()

    const lossValues = []
    const maeValues = []

    const fit_model_result = await fit_model.fit(xsNorm,ysNorm,{
        epochs : 250,
        callbacks : {
            onEpochEnd : async (epoch, logs) => {
                lossValues.push(logs.loss)
                maeValues.push(logs.mae)
            }
        }
    })

    await fit_model.save("file://./model")

    const loadedModel = await tf.loadLayersModel("file://./model/model.json")

    const inputRaw = tf.tensor2d([[35, 27, 9]])
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))

    const trust_analysis = await fit_model.evaluate(xsNorm,ysNorm)
    const [lossTensor,maeTensor] = trust_analysis // metrics mae dediğimiz için

    const predNorm = await loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)

    console.log("Son Kayıp: ",fit_model_result.history.loss.at(-1))
    console.log("Kayıp Analizi: ",lossTensor.dataSync()[0])
    console.log("Mutlak Sapma Analizi: ",maeTensor.dataSync()[0])

    pred.print()

    plot([{
        x : Array.from({length : lossValues.length},(_,i)=>i+1),
        y : lossValues,
        type : "line",
        name : "LOSS",
        line : {
            color : "red",
        }
    },{
        x : Array.from({length : maeValues.length},(_,i)=>i+1),
        y : maeValues,
        type : "line",
        name : "MAE",
        line : {
            color  : "green"
        }
    }],{
        xaxis : {
            title : "Epochs"
        },
        yaxis : {
            title : "Değerler"
        }
    })
}

trainAndPredict()
