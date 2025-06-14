// meme kanseri yakalanma durumunu sınıflandırma olarak analiz ettirelim

// kütüphaneler
const tf = require("@tensorflow/tfjs-node")
const Papa = require("papaparse")
const {plot} = require("nodeplotlib");
const fs = require("fs")

// dosya okuma kısımlarını gerceklestirelim
function readCsv(file_directory){
    const file = fs.readFileSync(file_directory, "utf8");

    const parsed = Papa.parse(file,{
        header : true,
        dynamicTyping : true,
        skipEmptyLines: true
    });

    const rows = parsed.data;

    const cleanedRows = rows.filter(row => {
        const values = Object.values(row);
        return values.every(val => val !== null && val !== "" && !isNaN(val) || row.diagnosis === "M" || row.diagnosis === "B");
    });

    const xsRaw = cleanedRows.map(row => [
        Number(row["radius_mean"]),
        Number(row["texture_mean"]),
        Number(row["perimeter_mean"]),
        Number(row["area_mean"]),
        Number(row["smoothness_mean"]),
        Number(row["compactness_mean"]),
        Number(row["concavity_mean"]),
        Number(row["concave_points_mean"]),
        Number(row["symmetry_mean"]),
        Number(row["fractal_dimension_mean"]),
        Number(row["radius_se"]),
        Number(row["texture_se"]),
        Number(row["perimeter_se"]),
        Number(row["area_se"]),
        Number(row["smoothness_se"]),
        Number(row["compactness_se"]),
        Number(row["concavity_se"]),
        Number(row["concave_points_se"]),
        Number(row["symmetry_se"]),
        Number(row["fractal_dimension_se"]),
        Number(row["radius_worst"]),
        Number(row["texture_worst"]),
        Number(row["perimeter_worst"]),
        Number(row["area_worst"]),
        Number(row["smoothness_worst"]),
        Number(row["compactness_worst"]),
        Number(row["concavity_worst"]),
        Number(row["concave_points_worst"]),
        Number(row["symmetry_worst"]),
        Number(row["fractal_dimension_worst"]),
    ]);

    const ysRaw = cleanedRows.map(row => [
        row["diagnosis"] === "M" ? 1 : 0,
    ]);

    return { xsRaw, ysRaw };
}


// girdi fonksiyonu
function readTestCsv(file_directory){
    const file = fs.readFileSync(file_directory, "utf8");

    const parsed = Papa.parse(file,{
        header : true,
        dynamicTyping : true,
        skipEmptyLines: true
    });

    const rows = parsed.data;

    const inputRaw = rows.map(row => [
        Number(row["radius_mean"]),
        Number(row["texture_mean"]),
        Number(row["perimeter_mean"]),
        Number(row["area_mean"]),
        Number(row["smoothness_mean"]),
        Number(row["compactness_mean"]),
        Number(row["concavity_mean"]),
        Number(row["concave_points_mean"]),
        Number(row["symmetry_mean"]),
        Number(row["fractal_dimension_mean"]),
        Number(row["radius_se"]),
        Number(row["texture_se"]),
        Number(row["perimeter_se"]),
        Number(row["area_se"]),
        Number(row["smoothness_se"]),
        Number(row["compactness_se"]),
        Number(row["concavity_se"]),
        Number(row["concave_points_se"]),
        Number(row["symmetry_se"]),
        Number(row["fractal_dimension_se"]),
        Number(row["radius_worst"]),
        Number(row["texture_worst"]),
        Number(row["perimeter_worst"]),
        Number(row["area_worst"]),
        Number(row["smoothness_worst"]),
        Number(row["compactness_worst"]),
        Number(row["concavity_worst"]),
        Number(row["concave_points_worst"]),
        Number(row["symmetry_worst"]),
        Number(row["fractal_dimension_worst"]),
    ]);

    return inputRaw;
}

const {xsRaw,ysRaw} = readCsv("train.csv")
const inputs = readTestCsv("test.csv")

// normalize fonksiyonu
function normalize(data){
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normalize = dataT.sub(min).div(max.sub(min)) // normalize fonksiyonu

    return {normalize,min,max}
}

const {normalize : xsNorm,min : xsMin, max : xsMax} = normalize(xsRaw)
const {normalize : ysNorm,min : ysMin, max : ysMax} = normalize(ysRaw)


// model oluşturalım
async function createModel(){
    const model = await tf.sequential()

    model.add(tf.layers.dense({
        units : 16,
        inputShape : [30],
        activation : "relu",
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.005
        })
    }))

    model.add(tf.layers.dropout({
        rate : 0.01
    }))

    model.add(tf.layers.dense({
        units : 7,
        activation : "relu",
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.005
        })
    }))

    model.add(tf.layers.dropout({
        rate : 0.01
    }))

    model.add(tf.layers.dense({
        units : 1,
        activation : "sigmoid",
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.005
        })
    }))

    model.compile({
        optimizer : tf.train.adam(0.001),
        loss : "binaryCrossentropy",
        metrics : ["mae","accuracy"]
    })

    return model
}

async function calculateR2(model, xsNorm, ysNorm, ysMin, ysMax) {
    // Model tahmini (normalize veri üzerinde)
    const predNorm = model.predict(xsNorm);

    // Asenkron şekilde verileri al
    const [predsDataNorm, actualDataNorm] = await Promise.all([predNorm.data(), ysNorm.data()]);

    // Denormalize et (min ve max'ı tek boyutlu array olarak alalım)
    const ysMinVal = ysMin.dataSync()[0];
    const ysMaxVal = ysMax.dataSync()[0];

    const predsData = predsDataNorm.map(v => v * (ysMaxVal - ysMinVal) + ysMinVal);
    const actualData = actualDataNorm.map(v => v * (ysMaxVal - ysMinVal) + ysMinVal);

    const mean = actualData.reduce((a, b) => a + b, 0) / actualData.length;

    const ssTot = actualData.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0);

    if (ssTot === 0) {
        console.error("Tüm gerçek değerler aynı, R² hesaplanamaz.");
        return NaN;
    }

    const ssRes = actualData.reduce((sum, val, i) => sum + Math.pow(val - predsData[i], 2), 0);

    const r2 = 1 - ssRes / ssTot;

    console.log(`📈 Eğitim Verisi Üzerinde R² Skoru: ${r2.toFixed(4)} (${(r2 * 100).toFixed(2)}%)`);

    return r2;
}

// model eğitme islemi
async function trainAndPredict(){
    const fit_model = await createModel()

    const lossValues = []
    const maeValues = []
    const accuracyValues = []

    const fit_model_result = await fit_model.fit(xsNorm,ysNorm,{
        epochs : 250,
        validationSplit: 0.2, // yüzde yirmisini validation olarak ayır. yoksa overfitting olabilir
        classWeight: {
            0: 1,  // B
            1: 3   // M → Daha fazla ağırlık ver  (Küçük sınıfı daha değerli hale getirelim)
        },
        callbacks : {
            onEpochEnd : (epoch, logs) => {
                lossValues.push(logs.loss);
                maeValues.push(logs.mae);
                accuracyValues.push(logs.acc || logs.accuracy)
            }
        }
    })

    await fit_model.save("file://./model")

    const loadedModel = await tf.loadLayersModel("file://./model/model.json")
    loadedModel.compile({
        optimizer : tf.train.adam(0.001),
        loss : "binaryCrossentropy",
        metrics : ["mae","accuracy"]
    })

    const inputRaw = tf.tensor2d(inputs);
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))

    const predNorm = loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)


    const [maeTensor,lossTensor,accTensor] = await loadedModel.evaluate(xsNorm,ysNorm)
    console.log("Son Model Kaybı: ",fit_model_result.history.loss.at(-1))
    console.log("Genel Kayıp: ",lossTensor.dataSync()[0])
    console.log("Genel Doğrulama: ",accTensor.dataSync()[0])

    // kayıp grafiği
    plot([{
        x : Array.from({length : lossValues.length},(_,i)=>i+1),
        y : lossValues,
        type : "line",
        name : "LOSS",
        line : {
            color : "blue"
        }
    }],{
        title : "Kayıp Grafiği (LOSS)",
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
        title : "Mutlak Hata Grafiği (MAE)",
        xaxis : {
            title : "Epochs"
        },
        yaxis : {
            title : "Değerler"
        }
    })

    // doğrulama kısmı grafiği
    plot([{
        x : Array.from({length : accuracyValues.length},(_,i)=>i+1),
        y : accuracyValues,
        type : "line",
        name : "ACC",
        line : {
            color : "purple"
        }
    }],{
        title : "Doğrulama Doğruluğu Grafiği (ACC)",
        xaxis : {
            title : "Epochs"
        },
        yaxis : {
            title : "Değerler"
        }
    })

    // Gerçek Tahmin Grafiği
    pred.data().then(predData=>{
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
                title : "Gerçek - Tahmin Grafiği (ActualPredict)",
                xaxis : {
                    title : "Gerçek Değerler"
                },
                yaxis : {
                    title : "Tahmin Değerler"
                }
            })
        })
    })

    // hata grafiği
    pred.data().then(predData=>{
        ysNorm.mul(ysMax.sub(ysMin)).add(ysMin).data().then(actualData=>{

            const errors = actualData.map((actual,i)=>actual - predData[i])

            plot([{
                x : Array.from(errors),
                type : "histogram",
                name : "Histogram",
                line : {
                    color : "red"
                }
            }],{
                title : "Hata Grafiği (Histogram)",
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
            for (let i=0; i<inputs.length; i++){
                console.log(`${i}. Girdi Sonucu: ${preds[i][0].toFixed(2)}`)
            }
        })
    })

    // R^2 hesaplama işlemi!!!
    await calculateR2(fit_model, xsNorm, ysNorm, ysMin, ysMax);
}

trainAndPredict()
