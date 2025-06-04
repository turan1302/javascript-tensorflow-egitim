/*

 */

const tf = require("@tensorflow/tfjs-node")
const {plot} = require("nodeplotlib");
const {model} = require("@tensorflow/tfjs-node");

const xsRaw = [
    [2, 1, 80],   // 2 oda, 1 banyo, 80 m²
    [3, 2, 120],
    [1, 1, 60],
    [4, 2, 150],
    [2, 1, 75],
    [3, 2, 110],
    [1, 1, 55],
    [5, 3, 200],
    [3, 1, 100],
    [2, 1, 70]
]

const ysRaw = [
    [500000],   // 500 bin TL
    [850000],
    [400000],
    [1200000],
    [480000],
    [780000],
    [390000],
    [1500000],
    [700000],
    [460000]
]

// normalize fonksiyonu yazalım
function normalize(data) {
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normalize = dataT.sub(min).div(max.sub(min))
    return {normalize, min, max}
}

// normalizasyon uygulayalım
const {normalize: xsNorm, min: xsMin, max: xsMax} = normalize(xsRaw)
const {normalize: ysNorm, min: ysMin, max: ysMax} = normalize(ysRaw)

// model oluşturma fonksiyonumuz
async function createModel() {
    const model = tf.sequential()
    model.add(tf.layers.dense({
        units: 16,
        inputShape: [3],
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({
            l2: 0.01
        })
    }))

    model.add(tf.layers.dropout({
        rate: 0.3
    }))

    model.add(tf.layers.dense({
        units: 8,
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({
            l2: 0.01
        })
    }))

    model.add(tf.layers.dropout({
        rate: 0.3
    }))

    model.add(tf.layers.dense({
        units: 4,
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({
            l2: 0.01
        })
    }))

    model.add(tf.layers.dropout({
        rate: 0.3
    }))

    model.add(tf.layers.dense({
        units: 2,
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({
            l2: 0.01
        })
    }))

    model.add(tf.layers.dropout({
        rate: 0.3
    }))

    model.add(tf.layers.dense({
        units: 1,
        activation: "linear",
    }))

    model.compile({
        optimizer: "adam",
        loss: "meanSquaredError"
    })

    return model
}

// model öğrenme ve tahmin kısmı
async function trainAndPredict() {
    const fit_model = await createModel()

    const lossValues = []

    const fit_model_result = await fit_model.fit(xsNorm, ysNorm, {
        epochs: 250,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                lossValues.push(logs.loss)
            }
        }
    })

    await fit_model.save("file://./model")

    const loadedModel = await tf.loadLayersModel("file://./model/model.json")

    const inpuRaw = tf.tensor2d([[3, 2, 110]])
    const inputNorm = inpuRaw.sub(xsMin).div(xsMax.sub(xsMin))

    const predNorm = loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)

    pred.print()


    const trust_analysis = await fit_model.evaluate(xsNorm,ysNorm)
    console.log("Son Kayıp: ",fit_model_result.history.loss.at(-1))
    console.log("Doğruluk: ",trust_analysis.dataSync()[0])


        // grafik görselleştirmesi
        plot([{
            x: Array.from({length: lossValues.length}, (_, i) => i + 1),
            y: lossValues,
            type: "line",
            name: "Loss"
        }], {
            title: "Kayıp Grafiği",
            yaxis: {
                title: "Kayıplar"
            },
            xaxis: {
                title: "Epochs"
            }
        })

}

trainAndPredict()
