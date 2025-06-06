/*
Makine Arıza Süresi Tahmini için
örnek verilerle birlikte basit bir TensorFlow.js regresyon modeli hazırlayalım.
 */

const tf = require("@tensorflow/tfjs-node")
const {plot} = require("nodeplotlib");

// Veri: [Sıcaklık, Titreşim, Çalışma Süresi, Son Bakım Zamanı]
const xsRaw = [
    [70, 3.5, 100, 10],
    [65, 2.8, 80, 20],
    [80, 4.0, 120, 5],
    [60, 3.0, 60, 15],
    [90, 5.0, 150, 3],
    [68, 3.2, 90, 12],
    [75, 4.1, 110, 8],
    [55, 2.5, 50, 25],
    [85, 4.8, 130, 4],
    [62, 3.0, 70, 18]
];

// Çıkış (Target): Kalan Arıza Süresi (saat)
const ysRaw = [
    [50],
    [75],
    [30],
    [90],
    [20],
    [60],
    [40],
    [100],
    [25],
    [80]
];

// normalize fonsiyonu
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

// model oluşturma, eğitme ve tahmin
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

    const inputRaw = tf.tensor2d(xsRaw)
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))

    const predNorm = loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)

    // doğrulıuk analizi
    const trust_analysis = await fit_model.evaluate(xsNorm,ysNorm)
    const [maeTensor,lossTensor]=trust_analysis

    console.log("Son Model Kaybı: ",fit_model_result.history.loss.at(-1))
    console.log("Doğruluk: ",lossTensor.dataSync()[0])
    pred.print()


    // loss table
    plot([{
        x : Array.from({length : lossValues.length},(_,i)=>i+1),
        y : lossValues,
        type : "line",
        name : "LOSS",
        line : {
            color : "gray"
        }
    }],{
        title : "Eğitim Kayıp Grafiği",
        xaxis : {
            title : "Epochs"
        },
        yaxis : {
            title : "Değerler"
        }
    })

    // mae (mutlak hata grafiği)
    plot([{
        x : Array.from({length : maeValues.length},(_,i)=>i+1),
        y : maeValues,
        type : "line",
        name : "MAE",
        line : {
            color : "blue"
        }
    }],{
        title : "Mutlak Kayıp Grafiği",
        xaxis : {
            title : "Epochs"
        },
        yaxis : {
            title : "Değerler"
        }
    })

    // gerçek ve tahmin kısmı
    pred.data().then(predData=>{
        ysNorm.mul(ysMax.sub(ysMin)).add(ysMin).data().then(actualData=>{
// gerçek - tahmin veri kısmı
            plot([{
                x : Array.from(actualData),
                y : Array.from(predData),
                type : "scatter",
                mode : "markers",
                name : "Pred - Actual",
                line : {
                    color : "green"
                }
            }],{
                title : "Gerçek - Tahmin Grafiği",
                xaxis : {
                    title : "Gerçek Değerler"
                },
                yaxis : {
                    title : "Tahmin Değerler"
                }
            })
        })
    })

    // veri tahmini
    inputRaw.array().then(inputs => {
        pred.array().then(preds => {
            for(let i=0;i<inputs.length;i++){
                console.log(`Girdi: ${inputs[i]} => Tahmin: ${preds[i][0].toFixed(2)} saat`);
            }
        })
    })
}

trainAndPredict()
