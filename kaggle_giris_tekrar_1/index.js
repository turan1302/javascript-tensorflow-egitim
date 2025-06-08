/*
burada kişinin sigorta ücretini tahmin edeceğiz
 */

const tf = require("@tensorflow/tfjs-node")
const {plot} = require("nodeplotlib");
const Papa = require("papaparse")
const fs = require("fs")

// eğitim verisi okuma işlemi
function readTrainCsv(file_directory) {
    const file = fs.readFileSync(file_directory, "utf8");

    const parsed = Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true
    });

    // Satırlardaki boş stringleri null yap
    const rows = parsed.data.map(row => {
        Object.keys(row).forEach(key => {
            if (typeof row[key] === "string") {
                row[key] = row[key].trim() === "" ? null : row[key];
            }
        });
        return row;
    });

    // Eksik veya NaN içeren satırları filtrele
    const filteredRows = rows.filter(row =>
        row.age !== null && !isNaN(Number(row.age)) &&
        row.sex !== null &&
        row.bmi !== null && !isNaN(Number(row.bmi)) &&
        row.children !== null && !isNaN(Number(row.children)) &&
        row.smoker !== null &&
        row.charges !== null && !isNaN(Number(row.charges))
    );

    const xsRaw = filteredRows.map(row => [
        Number(row.age),
        row.sex === "male" ? 1 : 0,
        Number(row.bmi),
        Number(row.children),
        row.smoker === "yes" ? 1 : 0
    ]);

    const ysRaw = filteredRows.map(row => [
        Number(row.charges)
    ]);

    // NaN kontrolü (debug amaçlı)
    if (xsRaw.some(row => row.some(isNaN))) {
        throw new Error("xsRaw içinde NaN var!");
    }
    if (ysRaw.some(row => row.some(isNaN))) {
        throw new Error("ysRaw içinde NaN var!");
    }

    return { xsRaw, ysRaw };
}

// test verisi okuma işlemi
function readTestCsv(file_directory) {
    const file = fs.readFileSync(file_directory, "utf8");

    const parsed = Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true
    });

    const rows = parsed.data.map(row => {
        Object.keys(row).forEach(key => {
            if (typeof row[key] === "string") {
                row[key] = row[key].trim() === "" ? null : row[key];
            }
        });
        return row;
    });

    const filteredRows = rows.filter(row =>
        row.age !== null && !isNaN(Number(row.age)) &&
        row.sex !== null &&
        row.bmi !== null && !isNaN(Number(row.bmi)) &&
        row.children !== null && !isNaN(Number(row.children)) &&
        row.smoker !== null
    );

    const inputRaw = filteredRows.map(row => [
        Number(row.age),
        row.sex === "male" ? 1 : 0,
        Number(row.bmi),
        Number(row.children),
        row.smoker === "yes" ? 1 : 0
    ]);

    // NaN kontrolü
    if (inputRaw.some(row => row.some(isNaN))) {
        throw new Error("Test inputRaw içinde NaN var!");
    }

    return inputRaw;
}

// verileri alalım
const {xsRaw,ysRaw} = readTrainCsv("train.csv")
const inputs = readTestCsv("test.csv")


// normalizasyon fonksiyonu
function normalize(data){
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normData = dataT.sub(min).div(max.sub(min)) // normalize işlemi

    return {normData,min,max}
}

// normalize işlemi
const {normData : xsNorm, min : xsMin, max : xsMax} = normalize(xsRaw)
const {normData : ysNorm, min : ysMin, max : ysMax} = normalize(ysRaw)

// model oluşturma işlemi
async function createModel(){
    const model = await tf.sequential()

    // giriş katmanı
    model.add(tf.layers.dense({
        units : 16,
        inputShape : [5],
        activation : "relu",
        kernelRegularizer : tf.regularizers.l2({l2 : 0.001})
    }))

    // gizli katman
    model.add(tf.layers.dropout({
        rate : 0.1
    }))

    // gizli katman
    model.add(tf.layers.dense({
        units : 7,
        activation : "relu",
    }))

    // gizli katman
    model.add(tf.layers.dropout({
        rate : 0.1
    }))

    // çıkış katmanı
    model.add(tf.layers.dense({
        units : 1,
        activation : "linear",
    }))

    // model derleyelim
    model.compile({
        optimizer : tf.train.adam(0.001),
        loss : "meanSquaredError",
        metrics : ["mae"]
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

// model eğitme ve tahmin kısmı
async function trainAndPredict(){
    const fit_model = await createModel()

    const maeValues = []
    const lossValues = []

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
        optimizer : tf.train.adam(0.001),
        loss : "meanSquaredError",
        metrics : ["mae"]
    })

    const inputRaw = tf.tensor2d(inputs)
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))


    const predNorm = loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)

    // kayıp grafiği çizdirelim
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
        title : "Mutlak Kayıp Grafiği (LOSS)",
        xaxis : {
            title : "Epochs"
        },
        yaxis : {
            title : "Değerler"
        }
    })

    // gerçek tahmin grafiği
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

    // histogram grfiği
    pred.data().then(predData=>{
        ysNorm.mul(ysMax.sub(ysMin)).add(ysMin).data().then(actualData=>{

            const errors = actualData.map((actual,i)=>actual - predData[i])

            plot([{
                x : Array.from(errors),
                type : "histogram",
                name : "Histogram",
                line : {
                    color : "blue"
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

    // trust kısmı
    const [maeTensor,lossTensor] = await loadedModel.evaluate(xsNorm,ysNorm)

    console.log("Son Model Kaybı: ",fit_model_result.history.loss.at(-1))
    console.log("Genel Kayıp: ",lossTensor.dataSync()[0])

    pred.print()

    // girdi ve tahmin analizi
    inputRaw.array().then(inputs=>{
        pred.array().then(preds=>{
            for (let i=0;i<inputs.length;i++){
                console.log(`${i}. Deneme ---> Girdi: ${inputs[i]} ==> Tahmin: ${preds[i][0].toFixed(2)} ₺`)
            }
        })
    })

    // R^2 hesaplama işlemi!!!
    await calculateR2(fit_model, xsNorm, ysNorm, ysMin, ysMax);
}

trainAndPredict()
