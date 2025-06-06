/*
Araba Yakıt Tüketimi Tahmini
Girişler:
[motor_hacmi, ağırlık_kg, beygir, üretim_yılı]

Çıkış (Target):
[litre/100km]
 */

const tf = require("@tensorflow/tfjs-node")
const {plot} = require("nodeplotlib");

// giriş verileri
const xsRaw = [
    [1.6, 1200, 110, 2018],
    [2.0, 1400, 150, 2020],
    [1.3, 1000, 90, 2016],
    [2.5, 1600, 180, 2021],
    [1.2, 950, 80, 2015],
    [1.8, 1300, 130, 2019],
    [1.4, 1100, 100, 2017],
    [3.0, 1800, 220, 2022],
    [2.2, 1500, 160, 2020],
    [1.5, 1150, 105, 2018]
]

// çıkış verileri
const ysRaw = [
    [6.2],
    [7.5],
    [5.8],
    [8.2],
    [5.5],
    [6.8],
    [6.0],
    [9.0],
    [7.2],
    [6.1]
]

// normalizasyon fonksiyonu
function normalize(data) {
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normalize = dataT.sub(min).div(max.sub(min)) // normalizasyon fonksiyonu

    return {normalize, min, max}
}

// normalizasyon uygulayalım
const {normalize: xsNorm, min: xsMin, max: xsMax} = normalize(xsRaw)
const {normalize: ysNorm, min: ysMin, max: ysMax} = normalize(ysRaw)

// model oluşturalım
async function createModel() {
    const model = await tf.sequential()

    // giriş katmanı
    model.add(tf.layers.dense({
        units: 16,
        inputShape: [4],
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({
            l2: 0.01
        })
    }))

    // gizli katman dropout
    model.add(tf.layers.dropout({
        rate: 0.01
    }))

    // gizli katman
    model.add(tf.layers.dense({
        units: 7,
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({
            l2: 0.01
        })
    }))

    // gizli katman dropout
    model.add(tf.layers.dropout({
        rate: 0.01
    }))

    // çıkış katmanı
    model.add(tf.layers.dense({
        units: 1,
        activation: "linear",
    }))

    // model derleyelim
    model.compile({
        optimizer: "adam",
        loss: "meanSquaredError",
        metrics: ["mae"]
    })

    return model
}

// model derleme ve eğitme
async function trainAndPredict() {
    const fit_model = await createModel()

    const lossValues = []
    const maeValues = []

    const fit_model_result = await fit_model.fit(xsNorm, ysNorm, {
        epochs: 250,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                lossValues.push(logs.loss)
                maeValues.push(logs.mae)
            }
        }
    })

    await fit_model.save("file://./model")

    const loadedModel = await tf.loadLayersModel("file://./model/model.json")

    loadedModel.compile({
        optimizer: "adam",
        loss: "meanSquaredError",
        metrics: ["mae"]
    })

    const inputRaw = tf.tensor2d([
        [1.7, 1250, 115, 2019],
        [1.3, 980, 85, 2016],
        [2.1, 1450, 155, 2021],
        [1.6, 1180, 108, 2018],
        [1.9, 1350, 140, 2020],
        [1.4, 1120, 95, 2017],
        [2.4, 1580, 175, 2021],
        [1.2, 960, 82, 2015],
        [3.2, 1900, 240, 2023],
        [1.5, 1100, 100, 2018]
    ])

    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))

    const predNorm = loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)

    // loss values kısmı
    plot([{
        x: Array.from({length: lossValues.length}, (_, i) => i + 1),
        y: lossValues,
        type: "line",
        name: "LOSS",
        line: {
            color: "green"
        }
    }], {
        title: "Kayıp Grafiği (LOSS)",
        xaxis: {
            title: "Epochs"
        },
        yaxis: {
            title: "Değerler"
        }
    })

    // mutlak hata grafiği
    plot([{
        x: Array.from({length: maeValues.length}, (_, i) => i + 1),
        y: maeValues,
        type: "line",
        name: "MAE",
    }], {
        title: "Mutlak Hata Grafiği (LOSS)",
        xaxis: {
            title: "Epochs"
        },
        yaxis: {
            title: "Değerler"
        }
    })

    // gerçek ve tahmin grafiği
    pred.data().then(predData=>{
        ysNorm.sub(ysMax.sub(ysMin)).add(ysMin).data().then(actualData=>{
            plot([{
                x: Array.from(actualData),
                y: Array.from(predData),
                type: "scatter",
                mode : "markers",
                name: "ActualPredict",
            }], {
                title: "Gerçek - Tahmin Grafiği (ActualPredict)",
                xaxis: {
                    title: "Gerçek"
                },
                yaxis: {
                    title: "Tahmin"
                }
            })
        })
    })

    // histogram kısmı
    pred.data().then(predData=>{
        ysNorm.sub(ysMax.sub(ysMin)).add(ysMin).data().then(actualData=>{

            const error = actualData.map((actual,i)=>actual - predData[i])

            plot([{
                x: Array.from(error),
                type: "histogram",
                name: "Histogram",
            }], {
                title: "Hata Histogram Grafiği (REH)",
                xaxis: {
                    title: "Hata"
                },
                yaxis: {
                    title: "Sıklık"
                }
            })
        })
    })

    inputRaw.array().then(inputs=>{
        pred.array().then(preds=>{
            for (let i=0;i<inputs.length;i++){
                console.log(`Girdi: ${inputs[i]} => Tahmin: ${preds[i][0].toFixed(2)} Litre`)
            }
        })
    })

    // doğruluk analizi
    const [lossTensor, maeTensor] = await loadedModel.evaluate(xsNorm, ysNorm)

    console.log("Son Model Kaybı: ", fit_model_result.history.loss.at(-1))
    console.log("Doğruluk: ", lossTensor.dataSync()[0])
    pred.print()
}

trainAndPredict()


