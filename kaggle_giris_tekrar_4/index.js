/*
kalp hastalığına yakalanma riskini oluşturalım
 */

const tf = require("@tensorflow/tfjs-node")
const {plot} = require("nodeplotlib");
const Papa = require("papaparse")
const fs = require("fs")

const encodeMap = {
    Sex: {M: 1, F: 0},
    ChestPainType: {ATA: 0, NAP: 1, ASY: 2, TA: 3}, // TA olmasa da diğer türler için map oluşturduk
    RestingECG: {Normal: 0, ST: 1, LVH: 2},
    ExerciseAngina: {N: 0, Y: 1},
    ST_Slope: {Up: 0, Flat: 1, Down: 2}
}

// read Csv fonksiyonunu oluşturalım
function readCsv(file_directory) {
    const file = fs.readFileSync(file_directory, "utf8")

    const parsed = Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true
    })

    const rows = parsed.data;

    const xsRaw = rows.map(row => [
        row.Age,
        encodeMap.Sex[row.Sex],
        encodeMap.ChestPainType[row.ChestPainType],
        row.RestingBP,
        row.Cholesterol,
        row.FastingBS,
        encodeMap.RestingECG[row.RestingECG],
        row.MaxHR,
        encodeMap.ExerciseAngina[row.ExerciseAngina],
        row.Oldpeak,
        encodeMap.ST_Slope[row.ST_Slope]
    ])

    const ysRaw = rows.map(row => [
        row.HeartDisease
    ])

    return {xsRaw, ysRaw}
}

// test verisi için de csv ayarı
function readTestCsv(file_directory) {
    const file = fs.readFileSync(file_directory, "utf8")

    const parsed = Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true
    })

    const rows = parsed.data;

    const inputRaw = rows.map(row => [
        row.Age,
        encodeMap.Sex[row.Sex],
        encodeMap.ChestPainType[row.ChestPainType],
        row.RestingBP,
        row.Cholesterol,
        row.FastingBS,
        encodeMap.RestingECG[row.RestingECG],
        row.MaxHR,
        encodeMap.ExerciseAngina[row.ExerciseAngina],
        row.Oldpeak,
        encodeMap.ST_Slope[row.ST_Slope]
    ])


    return inputRaw
}

const {xsRaw, ysRaw} = readCsv("train.csv")
const inputs = readTestCsv("test.csv")

// normalize fonksiyonu
function normalize(data) {
    const dataT = tf.tensor2d(data);
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normalize = dataT.sub(min).div(max.sub(min)) // normalize

    return {normalize, min, max}
}

const {normalize: xsNorm, min: xsMin, max: xsMax} = normalize(xsRaw)
const {normalize: ysNorm, min: ysMin, max: ysMax} = normalize(ysRaw)

// model oluşturalım
async function createModel() {
    const model = tf.sequential()

    model.add(tf.layers.dense({
        units: 34,
        inputShape: [11],
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({
            l2: 0.001
        })
    }))

    model.add(tf.layers.dropout({
        rate: 0.01
    }))

    model.add(tf.layers.dense({
        units: 13,
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({
            l2: 0.001
        })
    }))

    model.add(tf.layers.dropout({
        rate: 0.01
    }))

    model.add(tf.layers.dense({
        units: 8,
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({
            l2: 0.001
        })
    }))


    model.add(tf.layers.dropout({
        rate: 0.01
    }))

    model.add(tf.layers.dense({
        units: 1,
        activation: "sigmoid",
        kernelRegularizer: tf.regularizers.l2({
            l2: 0.001
        })
    }))

    model.compile({
        optimizer: tf.train.adam(0.005),
        loss: "binaryCrossentropy",
        metrics: ["accuracy", "mae"]
    })

    return model;
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


// model eğitme ve tahmin yaptırma
async function trainAndPredict() {
    const fit_model = await createModel()

    const lossValues = [];
    const maeValues = [];
    const accValues = [];

    const fit_model_result = await fit_model.fit(xsNorm, ysNorm, {
        epochs: 250,
        validationSplit: 0.2, // overfit olmasın diye verilerin yüzde 20 sini validation için ayarladık
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                lossValues.push(logs.loss);
                maeValues.push(logs.mae);
                accValues.push(logs.acc || logs.accuracy)
            }
        }
    })

    await fit_model.save("file://./model")
    const loadedModel = await tf.loadLayersModel("file://./model/model.json")
    loadedModel.compile({
        optimizer: tf.train.adam(0.005),
        loss: "binaryCrossentropy",
        metrics: ["accuracy", "mae"]
    })

    const inputRaw = tf.tensor2d(inputs);
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin));
    const predNorm = loadedModel.predict(inputNorm);
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin);

    pred.print();

    const [lossTensor, maeTensor, accTensor] = await loadedModel.evaluate(xsNorm, ysNorm)
    console.log("Son Model Kaybı: ", fit_model_result.history.loss.at(-1))
    console.log("Genel Kayıp: ", lossTensor.dataSync()[0])
    console.log("Genel Mutlak Hata Kaybı: ", maeTensor.dataSync()[0])
    console.log("Genel Doğrulama Kaybı: ", accTensor.dataSync()[0])

    // model kaybı grafiği
    plot([{
        x : Array.from({length : lossValues.length},(_,i)=>i),
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

    // mutlak hata kaybı grafiği
    plot([{
        x : Array.from({length : maeValues.length},(_,i)=>i),
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

    // gerçek - tahmin grafiği
    pred.data().then(predData=>{
        ysNorm.mul(ysMax.sub(ysMin)).add(ysMin).data().then(actualData=>{
            plot([{
                x : Array.from(actualData),
                y : Array.from(predData),
                type : "scatter",
                mode : "markers",
                name : "ActualPredict",
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

    // hata grafiği (histogram)
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

    // accuracy hesabı
    const preds = loadedModel.predict(xsNorm);
    const predsBinary = preds.greater(0.5).cast("float32");
    const accuracy = predsBinary.equal(ysNorm).mean().dataSync()[0];
    console.log("Accuracy (thresholded): ", accuracy);

    inputRaw.array().then(inputs => {
        pred.array().then(preds => {
            for (let i = 0; i < inputs.length; i++) {
                console.log(`${i}. Girdi Sonucu: ${preds[i][0].toFixed(2)}`)
            }
        })
    });

    await calculateR2(fit_model, xsNorm, ysNorm, ysMin, ysMax);
}

trainAndPredict()
