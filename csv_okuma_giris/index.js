/*
bu kısımda csv verilerini okuma işlemlerini gerçekleştircez

papaparse kütüphanesini kullanacağız
 */


const tf = require("@tensorflow/tfjs-node")
const {plot} = require("nodeplotlib");
const Papa = require("papaparse")
const fs = require("fs")

// csv verisini okuyalım
function readCsvFile(file_directory) {
    const file = fs.readFileSync(file_directory, "utf8")

    const parsed = Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true
    })

    const rows = parsed.data;

    const xsRaw = rows.map(row => [
        row.motor_hacmi,
        row.ağırlık_kg,
        row.beygir,
        row.üretim_yılı
    ])

    const ysRaw = rows.map(row => [
        row.litre_100km
    ])


    return {xsRaw, ysRaw}
}

const {xsRaw, ysRaw} = readCsvFile("veri.csv")


// normalizasyon fonksiyonu
function normalize(data){
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normalize = dataT.sub(min).div(max.sub(min)) // normalize fonksiyonunu yazalım

    return {normalize,min,max}
}

// normalize uygulayalım
const {normalize : xsNorm, min : xsMin, max : xsMax} = normalize(xsRaw)
const {normalize : ysNorm, min : ysMin, max : ysMax} = normalize(ysRaw)

// model oluşturma fonksiyonu
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
        rate : 0.01
    }))

    // gizli katman
    model.add(tf.layers.dense({
        units : 9,
        activation : "relu",
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.01
        })
    }))

    // gizli katman dropout
    model.add(tf.layers.dropout({
        rate : 0.01
    }))

    // çıkış katmanı
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

// model eğitimi ve tahmin
async function trainAndPredict(){
    const fit_model = await createModel()

    const lossValues = []
    const maeValues = []

    const fit_model_result = await fit_model.fit(xsNorm,ysNorm,{
        epochs : 250,
        callbacks : {
            onEpochEnd : (epoch, logs) => {
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

    // veri eğitim kısımları
    const inputRaw = tf.tensor2d([
        [1.7, 1250, 115, 2019],
        [2.0, 1400, 140, 2021],
        [1.3, 980, 88, 2016],
    ])
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))

    const predNorm = loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)

    // kayıp grafiği
    plot([{
        x : Array.from({length : lossValues.length},(_,i)=>i+1),
        y : lossValues,
        type : "line",
        name : "LOSS",
        line : {
            color : "gray"
        }
    }],{
        title : "Kayıp Grafiği (Loss)",
        xaxis : {
            title : "Epochs"
        },
        yaxis : {
            title : "Değerler"
        }
    })

    // mutlak hata grafiği
    plot([{
        x : Array.from({length : maeValues.length},(_,i)=>i+1),
        y : maeValues,
        type : "line",
        name : "MAE",
        line : {
            color : "orange"
        }
    }],{
        title : "Mutlak Kayıp Grafiği (MAE)",
        xaxis : {
            title : "Epochs"
        },
        yaxis : {
            title : "Değerler"
        }
    })

    // gerçek - tahmin değerler
    pred.data().then(predData=>[
        ysNorm.mul(ysMax.sub(ysMin)).add(ysMin).data().then(actualData=>{
            plot([{
                x : Array.from(actualData),
                y : Array.from(predData),
                type : "scatter",
                mode : "markers",
                name : "ActualPred",
                line : {
                    color : "green"
                }
            }],{
                title : "Tahmin - Gerçek Değerler",
                xaxis : {
                    title : "Gerçek Değerler"
                },
                yaxis : {
                    title : "Tahmin Değerler"
                }
            })
        })
    ])

    // hata histogramı
    pred.data().then(predData=>[
        ysNorm.mul(ysMax.sub(ysMin)).add(ysMin).data().then(actualData=>{

            const errors = actualData.map((actual,i)=>actual - predData[i])

            plot([{
                x : Array.from(errors),
                type : "histogram",
                name : "Histogram",
                line : {
                    color : "green"
                }
            }],{
                title: "Hata Histogram Grafiği (REH)",
                xaxis : {
                    title : "Hatalar"
                },
                yaxis : {
                    title : "Sıklık"
                }
            })
        })
    ])

    inputRaw.array().then(inputs=>{
        pred.array().then(preds=>{
            for (let i=0;i<inputs.length;i++){
                console.log(`Girdi: ${inputs[i]} => Tahmin: ${preds[i][0].toFixed(2)} Lt`)
            }
        })
    })

    // dorğuluk analizi
    const [maeTensor,lossTensor] = await loadedModel.evaluate(xsNorm,ysNorm)

    console.log("Son Model Kaybı: ",fit_model_result.history.loss.at(-1))
    console.log("Doğruluk: ",lossTensor.dataSync()[0])

    pred.print()

}

trainAndPredict()
