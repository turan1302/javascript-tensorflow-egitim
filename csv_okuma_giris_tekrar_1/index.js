/*
Akciğer Kanseri Yakalanma Riski Analizi

Bu CSV dosyasında şu kolonlar olacak:

| yaş | sigara_içiyor_mu | yıllık_sigara_adeti | ailede_kanser_öyküsü | maruz_kaldığı_hava_kirliliği | kanser_olasiligi |

yaş: 30 ile 85 arasında tam sayı

sigara_içiyor_mu: 0 (hayır) veya 1 (evet)

yıllık_sigara_adeti: sigara içiyorsa yılda içilen sigara adeti (5,000 ile 25,000 arasında), içmiyorsa 0

ailede_kanser_öyküsü: 0 veya 1 (var mı yok mu)

maruz_kaldığı_hava_kirliliği: 1 ile 10 arası, 10 en kötü

kanser_olasiligi: 0 ile 1 arası, kabaca yukarıdaki değerlerden üretilmiş gerçekçi bir tahmin olasılık değeri
 */

const tf = require("@tensorflow/tfjs-node")
const {plot} = require("nodeplotlib");
const Papa = require("papaparse")
const fs = require("fs")

// veri setini dahil edelim
function readCsvFile(file_directory){
    const file = fs.readFileSync(file_directory,"utf8")

    const parsed = Papa.parse(file,{
        header : true,
        dynamicTyping : true,
        skipEmptyLines: true
    })

    const rows = parsed.data

    const xsRaw = rows.map(row=>[
        row.yaş,
        row.sigara_içiyor_mu,
        row.yıllık_sigara_adeti,
        row.ailede_kanser_öyküsü,
        row.maruz_kaldığı_hava_kirliliği
    ])

    const ysRaw = rows.map(row=>[
        row.kanser_olasiligi
    ])

    return {xsRaw,ysRaw}
}

// değerleri alalım
const {xsRaw,ysRaw} = readCsvFile("veri.csv")

// normalizasyon fonksiyonu oluşturalım
function normalize(data){
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normalize = dataT.sub(min).div(max.sub(min)) // normalize formülü

    return {normalize,min,max}
}

// normalize uygulayalım
const {normalize : xsNorm, min : xsMin, max : xsMax} = normalize(xsRaw)
const {normalize : ysNorm, min : ysMin, max : ysMax} = normalize(ysRaw)

// model oluşturma işlemi
async function createModel(){
    const model = tf.sequential()

    // giriş katmanı
    model.add(tf.layers.dense({
        units : 16,
        inputShape : [5],
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
        units : 7,
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

    // model compile
    model.compile({
        optimizer : "adam",
        loss : "meanSquaredError",
        metrics : ["mae"]
    })

    return model
}

// model eğitme ve tahmin kısmı
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

    // eğitim kısmı olayı
    const inputRaw = tf.tensor2d( [
        [55, 1, 15000, 1, 7],
        [48, 0, 0, 0, 4],
        [62, 1, 18000, 1, 8],
        [37, 0, 0, 0, 3],
        [53, 1, 12000, 1, 6],
        [35, 0, 0, 0, 2],
        [57, 1, 17000, 1, 7],
        [49, 0, 0, 0, 4],
        [63, 1, 21000, 1, 9],
        [42, 0, 0, 0, 3],
        [55, 1, 14000, 1, 6],
        [38, 0, 0, 0, 3],
        [61, 1, 19000, 1, 8],
        [47, 0, 0, 0, 5],
        [54, 1, 15000, 1, 7]
    ])
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))
    const predNorm = loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)


    // doğruluk analizi
    const [lossTensor,maeTensor] = await loadedModel.evaluate(xsNorm,ysNorm)

    console.log("Son Model Kaybı: ",fit_model_result.history.loss.at(-1))
    console.log("Doğruluk: ",lossTensor.dataSync()[0])

    // loss grafiği
    plot([{
        x : Array.from({length : lossValues.length},(_,i)=>i+1),
        y : lossValues,
        type : "line",
        name : "LOSS",
        line : {
            color : "gray"
        }
    }],{
        title : "Loss Grafiği",
        xaxis : {
            title : "Epochs"
        },
        yaxis :  {
            title : "Değerler"
        }
    })

    // mutlak kata grafiği
    plot([{
        x : Array.from({length : maeValues.length},(_,i)=>i+1),
        y : maeValues,
        type : "line",
        name : "MAE",
        line : {
            color : "orange"
        }
    }],{
        title : "Mutlak Hata Grafiği (MAE)",
        xaxis : {
            title : "Epochs"
        },
        yaxis :  {
            title : "Değerler"
        }
    })

    // gerçek - tahmin değeri
    pred.data().then(predsData =>{
        ysNorm.mul(ysMax.sub(ysMin)).add(ysMin).data().then(actualData=>{
            plot([{
                x : Array.from(actualData),
                y : Array.from(predsData),
                type : "scatter",
                mode : "markers",
                name : "actualPreds",
                line : {
                    color : "green"
                }
            }],{
                title : "Gerçek - Tahmin Grafiği",
                xaxis : {
                    title : "Gerçek"
                },
                yaxis : {
                    title : "Tahmin"
                }
            })
        })
    })

    // histogram kısmı (hata sıklığı)
    pred.data().then(predsData =>{
        ysNorm.mul(ysMax.sub(ysMin)).add(ysMin).data().then(actualData=>{

            const errors = actualData.map((actual,i)=>actual - predsData[i])

            plot([{
                x : Array.from(errors),
                type : "histogram",
                name : "histogram",
                line : {
                    color : "blue"
                }
            }],{
                title : "Histogram Grafiği",
                xaxis : {
                    title : "Hata"
                },
                yaxis : {
                    title : "Sıklık"
                }
            })
        })
    })


    inputRaw.array().then(inputs=>{
        pred.array().then(preds=>{
            for (let i=0;i<inputs.length;i++){
                console.log(`Girdi: ${inputs[i]} => Tahmin: ${preds[i][0].toFixed()} (0 - 1 Arası)`)
            }
        })
    })

}

trainAndPredict()
