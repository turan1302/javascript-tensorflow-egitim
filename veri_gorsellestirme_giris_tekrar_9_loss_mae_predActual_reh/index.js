/*
1. Ev Fiyat Tahmini - Test Verileri
Giriş Formatı: [metrekare, oda_sayısı, bina_yaşı, semt_puanı]
Beklenen Çıkış: [fiyat_TL]
 */

const tf = require("@tensorflow/tfjs-node")
const {plot} = require("nodeplotlib");

// giriş verileri
const xsRaw = [
    [90, 2, 5, 70],
    [120, 3, 2, 85],
    [75, 2, 20, 60],
    [150, 4, 1, 90],
    [60, 1, 25, 50],
    [100, 3, 10, 80],
    [85, 2, 15, 65],
    [130, 3, 3, 88],
    [95, 2, 12, 72],
    [110, 3, 5, 78]
]

// çıkış verileri
const ysRaw = [
    [950000],
    [1500000],
    [720000],
    [2000000],
    [600000],
    [1300000],
    [800000],
    [1750000],
    [1100000],
    [1400000]
]

// normalizasyon fonksiyonu
function normalize(data){
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // aixs boyunca max
    const normalize = dataT.sub(min).div(max.sub(min)) // normalizasyon formülü

    return {normalize, min, max}
}

// normalizasyon uygylayalım
const {normalize : xsNorm, min : xsMin, max : xsMax} = normalize(xsRaw)
const {normalize : ysNorm, min : ysMin, max : ysMax} = normalize(ysRaw)

// model oluşturma
async function createModel(){
    const model = await tf.sequential()

    // giriş katmanı
    model.add(tf.layers.dense({
        units : 16,
        inputShape : [4],
        activation : "relu",
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.01
        })
    }))

    // gizli katman dropout
    model.add(tf.layers.dropout({
        rate : 0.1
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
        rate : 0.1
    }))

    // çıkış katmanı
    model.add(tf.layers.dense({
        units : 1,
        activation : "linear",
    }))

    // mdel derleyelim
    model.compile({
        optimizer : "adam",
        loss : "meanSquaredError",
        metrics : ["mae"]
    })

    return model
}

// model eğitelim ve tahmin yaptıralım
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

    loadedModel.compile({
        optimizer : "adam",
        loss : "meanSquaredError",
        metrics : ["mae"]
    })

    const inputRaw = tf.tensor2d([
        [105, 3, 8, 75],
        [70, 2, 18, 60],
        [140, 4, 3, 88],
        [95, 2, 6, 68],
        [80, 2, 14, 55],
        [115, 3, 5, 82],
        [65, 1, 20, 52],
        [125, 3, 2, 85],
        [90, 2, 10, 70],
        [100, 3, 12, 77]
    ])

    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))

    const predNorm = loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)


    // doğruluk
    const trust_analysis = await loadedModel.evaluate(xsNorm,ysNorm)
    const [maeTensor,lossTensor] = trust_analysis;

    // loss grafiği
    plot([{
        x : Array.from({length : lossValues.length},(_,i)=>i+1),
        y : lossValues,
        type : "line",
        name : "LOSS"
    }],{
        title : "Kayıp Grafiği (Loss)",
        xaxis : {
            title : "Epochs"
        },
        yaxis : {
            title : "Değerler"
        }
    })

    // mae grafiği
    plot([{
        x : Array.from({length : maeValues.length},(_,i)=>i+1),
        y : maeValues,
        type : "line",
        name : "MAE",
        line : {
            color : "black"
        }
    }],{
        title : "Mutlak Hata Grafiği (Loss)",
        xaxis : {
            title : "Epochs"
        },
        yaxis : {
            title : "Değerler"
        }
    })

    // gerçek, tahmin değerler
    pred.data().then(predData=>{
        ysNorm.mul(ysMax.sub(ysMin)).add(ysMin).data().then(actualData =>{
            plot([{
                x : Array.from(actualData),
                y : Array.from(predData),
                type : "scatter",
                mode : "markers",
                name : "PredActual",
                line : {
                    color : "black"
                }
            }],{
                title : "Gerçek - Tahmin Tablosu (Pred - Actual)",
                xaxis : {
                    title : "Gerçek Değerler"
                },
                yaxis : {
                    title : "Tahmin Değerler"
                }
            })
        })
    })

    // hata DAĞILIMI
    pred.data().then(predData=>{
        ysNorm.mul(ysMax.sub(ysMin)).add(ysMin).data().then(actualData =>{

            const errors = actualData.map((actual,i)=>actual - predData[i])
            console.log("------------")
            console.log(errors)
            console.log("------------")

            plot([{
                x : Array.from(errors),
                type : "histogram",
                name : "Hatalar",
                line : {
                    color : "black"
                }
            }],{
                title : "Hata Dağılımı (Histogram)",
                xaxis: { title: "Hata Miktarı (Gerçek - Tahmin)" },
                yaxis: { title: "Sıklık" }
            })
        })
    })


    inputRaw.array().then(inputs=>{
        pred.array().then(preds=>{
            for(let i=0;i<inputs.length;i++){
                console.log(`Girdi: ${inputs[i]} => Tahmin ${preds[i][0].toFixed(2)} TL`)
            }
        })
    })

    console.log("Son Model Kaybı: ",fit_model_result.history.loss.at(-1))
    console.log("Doğruluk: ",lossTensor.dataSync()[0])

    pred.print()
}

trainAndPredict()
