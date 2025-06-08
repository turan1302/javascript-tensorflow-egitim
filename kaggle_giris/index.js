/*
    kaggle veri setleri ile çalışma işlemine girelim
 */

const tf = require("@tensorflow/tfjs-node")
const {plot} = require("nodeplotlib");
const Papa = require("papaparse")
const fs = require("fs")

// eğitim model okuyucu
function readTrainCsv(file_directory){
    const file = fs.readFileSync(file_directory,"utf8")

    const parsed = Papa.parse(file,{
        header : true,
        dynamicTyping : true,
        skipEmptyLines: true
    })

    const rows = parsed.data

    const filteredRows = rows.filter(row =>
        row.Age != null &&
        row.Fare != null &&
        row.Sex != null &&
        row.Embarked != null
    )

    const xsRaw = filteredRows.map(row => [
        row.Pclass,
        row.Sex === "male" ? 1 : 0,
        row.Age,
        row.SibSp,
        row.Parch,
        row.Fare,
        row.Embarked === "S" ? 0 :
            row.Embarked === "C" ? 1 :
                row.Embarked === "Q" ? 2 : -1
    ])

    const ysRaw = filteredRows.map(row => [row.Survived])

    return {xsRaw,ysRaw}
}

// test modeli okuyucu
function readTestCsv(file_directory){
    const file = fs.readFileSync(file_directory,"utf8")

    const parsed = Papa.parse(file,{
        header : true,
        dynamicTyping : true,
        skipEmptyLines: true
    })

    const rows = parsed.data

    const filteredRows = rows.filter(row =>
        row.Age != null &&
        row.Fare != null &&
        row.Sex != null &&
        row.Embarked != null
    )

    const inputRaw = filteredRows.map(row => [
        row.Pclass,
        row.Sex === "male" ? 1 : 0,
        row.Age,
        row.SibSp,
        row.Parch,
        row.Fare,
        row.Embarked === "S" ? 0 :
            row.Embarked === "C" ? 1 :
                row.Embarked === "Q" ? 2 : -1
    ])

    return inputRaw
}

// verileri alalım
const {xsRaw,ysRaw} = readTrainCsv("train.csv")
const inputs = readTestCsv("train.csv")

// normalizasyon fonksiyonu
function normalize(data){
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normalize = dataT.sub(min).div(max.sub(min)) // normalize fonksiyonu

    return {normalize,min,max}
}

// normalizasyon uygulayalım
const {normalize : xsNorm, min : xsMin, max : xsMax} = normalize(xsRaw)
const {normalize : ysNorm, min : ysMin, max : ysMax} = normalize(ysRaw)

// model oluşturalım
async function createModel(){
    const model = tf.sequential()

    model.add(tf.layers.dense({
        units: 64,
        inputShape: [7],
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
    }));

    model.add(tf.layers.dropout({ rate: 0.1 }));

    model.add(tf.layers.dense({
        units: 32,
        activation: 'relu'
    }));

    model.add(tf.layers.dropout({ rate: 0.1 }));

    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));

    model.compile({
        optimizer : tf.train.adam(0.001), // Daha düşük öğrenme oranı
        loss : "binaryCrossentropy",
        metrics : ["accuracy"] // mae yerine accuracy
    })

    return model
}

// model eğitme ve tahöin ettirme
async function trainAndPredict(){
    const fit_model = await createModel()

    const maeValues = []
    const lossValues = []
    const accuracyValues = []

    const fit_model_result = await fit_model.fit(xsNorm,ysNorm,{
        epochs : 250,
        callbacks : {
            onEpochEnd : async (epoch, logs) => {
                lossValues.push(logs.loss)
                maeValues.push(logs.mae)
                accuracyValues.push(logs.acc || logs.accuracy)
            }
        }
    })

    await fit_model.save("file://./model")

    const loadedModel = await tf.loadLayersModel("file://./model/model.json")

    loadedModel.compile({
        optimizer : "adam",
        loss : "binaryCrossentropy",  // çıkış katmanı sigmoid olunca bunu da bu şekilde ayarladık
        metrics : ["mae"]
    })

    const inputRaw = tf.tensor2d(inputs)
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))
    const predNorm = loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)

    // kayıp grafiği
    plot([{
        x : Array.from({length : lossValues.length},(_,i)=>i+1),
        y : lossValues,
        name : "LOSS",
        type : "line",
        line : {
            color : "gray"
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
        x : Array.from({length : accuracyValues.length},(_,i)=>i+1),
        y : accuracyValues,
        name : "MAE",
        type : "line",
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

    // gerçek ve tahmin kısmı
    pred.data().then(predData=>{
        ysNorm.mul(ysMax.sub(ysMin)).add(ysMin).data().then(actualData=>{
            plot([{
                x : Array.from(actualData),
                y : Array.from(predData),
                name : "ActualPredict",
                type : "scatter",
                mode : "markers",
                line : {
                    color : "green"
                }
            }],{
                title : "Gerçek - Tahmin Grafiği",
                xaxis : {
                    title : "Gerçek Veriler"
                },
                yaxis : {
                    title : "Tahmin Veriler"
                }
            })
        })
    })

    // histogram grafiği
    pred.data().then(predData=>{
        ysNorm.mul(ysMax.sub(ysMin)).add(ysMin).data().then(actualData=>{

            const errors = actualData.map((actual,i)=>actual - predData[i])

            plot([{
                x : Array.from(errors),
                name : "Histogram",
                type : "histogram",
                line : {
                    color : "blue"
                }
            }],{
                title : "Hata Histogram Grafiği (REH)",
                xaxis : {
                    title : "Hata"
                },
                yaxis : {
                    title : "Sıklık"
                }
            })
        })
    })

    // turst analysis
    const [maeTensor,lossTensor] = await loadedModel.evaluate(xsNorm,ysNorm)

    console.log("Son Model Kaybı: ",fit_model_result.history.loss.at(-1))
    console.log("Doğruluk: ",lossTensor.dataSync()[0])
    pred.print()

    inputRaw.array().then(inputs=>{
        pred.array().then(preds=>{
            for (let i=0;i<inputs.length;i++){
                console.log(` ${i}. deneme--> Girdi: ${inputs[i]} => Tahmin: ${preds[i][0].toFixed(2)} İhtimal`)
            }
        })
    })

    // R^2 hesaplama işlemi!!!
    Promise.all([predNorm.data(), ysNorm.data()]).then(([predsDataNorm, actualDataNorm]) => {
        // Denormalize et
        const predsData = predsDataNorm.map(
            (v) => v * (ysMax.arraySync()[0] - ysMin.arraySync()[0]) + ysMin.arraySync()[0]
        );
        const actualData = actualDataNorm.map(
            (v) => v * (ysMax.arraySync()[0] - ysMin.arraySync()[0]) + ysMin.arraySync()[0]
        );

        const mean = actualData.reduce((a, b) => a + b, 0) / actualData.length;
        const ssTot = actualData.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0);

        if (ssTot === 0) {
            console.error("Tüm gerçek değerler aynı, R² hesaplanamaz.");
            return;
        }

        const ssRes = actualData.reduce((sum, val, i) => sum + Math.pow(val - predsData[i], 2), 0);
        const r2 = 1 - ssRes / ssTot;

        console.log(`📈 Tüm Veride R² Skoru: ${r2.toFixed(4)} (${(r2 * 100).toFixed(2)}%)`);
    });
}

trainAndPredict()
