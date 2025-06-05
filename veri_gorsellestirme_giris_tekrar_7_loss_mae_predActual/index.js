/*
💡 Örnek Senaryo:
Günlük Kahve Tüketimi (bardak) + Uyku Süresi (saat) → Günlük Odaklanma Skoru (0–100)

🎯 Açıklama:
Çok kahve odaklanmayı bir yere kadar artırır ama fazla içince düşebilir.

Uykusuzsan düşük, iyi uyuduysan yüksek olur.

Dengeli uyku + makul kahve = yüksek odaklanma.
 */

const tf = require("@tensorflow/tfjs-node")
const {plot} = require("nodeplotlib");

// giriş verileri
const xsRaw = [
    [1, 4],   // az kahve, az uyku
    [2, 5],
    [3, 6],
    [4, 6.5],
    [5, 7],
    [6, 7.5],
    [3, 8],   // ideal uyku, orta kahve
    [2, 8.5],
    [1, 9],
    [5, 5],   // çok kahve, az uyku
    [6, 4],
    [4, 8],   // yüksek kahve ama iyi uyku
    [0, 8],   // kahvesiz ama iyi uyku
    [3, 7.5],
    [2, 6],
    [1, 5.5],
    [4, 4.5],
    [5, 6.5],
    [6, 6],
    [2, 7]
]

// çıkış verileri
const ysRaw = [
    [40],
    [50],
    [60],
    [68],
    [75],
    [78],
    [85],
    [80],
    [76],
    [55],
    [45],
    [82],
    [78],
    [79],
    [65],
    [55],
    [50],
    [72],
    [70],
    [74]
]

// normalizasyon fonlsiyonu
function normalize(data){
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normalize = dataT.sub(min).div(max.sub(min)) // normalize formülü

    return {normalize,min,max}
}

// normalizasyon uygulayalım
const {normalize : xsNorm, min : xsMin, max : xsMax} = normalize(xsRaw)
const {normalize : ysNorm, min : ysMin, max : ysMax} = normalize(ysRaw)

// model oluşturma fonksiyonu
async function createModel(){
    const model = await tf.sequential()

    // giriş katmanı
    model.add(tf.layers.dense({
        units : 16,
        inputShape : [2],
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

// model eğitme ve tahmin yaptırma
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

    // eğitim işlemleri
    const inputRaw = tf.tensor2d(xsRaw)
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))

    const predNorm = loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)

    // lossValues çizdirelim
    plot([{
        x : Array.from({length : lossValues.length},(_,i)=>i+1),
        y : lossValues,
        type : "line",
        name : "LOSS",
        line : {
            color : "black"
        }
    }],{
        title : "Kayıp Grafiği",
        yaxis : {
            title : "Değerler"
        },
        xaxis : {
            title : "Epochs"
        }
    })

    // lossValues çizdirelim
    plot([{
        x : Array.from({length : maeValues.length},(_,i)=>i+1),
        y : maeValues,
        type : "line",
        name : "MAE",
        line : {
            color : "green",
        }
    }],{
        title : "Mutlak Sapma Grafiği",
        yaxis : {
            title : "Değerler"
        },
        xaxis : {
            title : "Epochs"
        }
    })


    // gerçek ve tahmin grafiği
    pred.data().then(predData =>{
        ysNorm.mul(ysMax.sub(ysMin)).add(ysMin).data().then(actualData =>{
            plot([{
                x : Array.from(actualData),
                y : Array.from(predData),
                type : "scatter",
                mode : "markers",
                name : "predActual",
            }],{
                title : "Gerçek - Tahmin Grafiği",
                yaxis : {
                    title : "Tahmin"
                },
                xaxis : {
                    title : "Gerçek Değerler"
                }
            })
        })
    })

    const trust_analysis = await fit_model.evaluate(xsNorm,ysNorm)
    const [maeTensor,lossTensor] = trust_analysis;

    // model kodları
    console.log("Son Kayıp: ",fit_model_result.history.loss.at(-1))
    console.log("Doğruluk: ",lossTensor.dataSync()[0])
    pred.print()

}

trainAndPredict()
