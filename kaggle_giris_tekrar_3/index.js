// iris dataset işlemini kodlayalım

const tf = require("@tensorflow/tfjs-node")
const {plot} = require("nodeplotlib");
const Papa = require("papaparse")
const fs = require("fs")

// eğitim verisini okuyalım
let speciesList = []
function readCsv(file_directory) {
    const file = fs.readFileSync(file_directory, "utf8")

    const parsed = Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true
    })

    const rows = parsed.data;

    const xsRaw = rows.map(row => [
        Number(row.SepalLengthCm),
        Number(row.SepalWidthCm),
        Number(row.PetalLengthCm),
        Number(row.PetalWidthCm),
    ])

    // Unique türleri bulalım
    const speciesSet = new Set(rows.map(row => row.Species))
    speciesList = Array.from(speciesSet)

    // One-hot encode fonksiyonu
    const oneHotEncode = (species) => {
        return speciesList.map(s => (s === species ? 1 : 0))
    }

    const ysRaw = rows.map(row => oneHotEncode(row.Species))

    return { xsRaw, ysRaw }
}

// test verisinmi okuyalım
function readTestCsv(file_directory) {
    const file = fs.readFileSync(file_directory, "utf8")

    const parsed = Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true
    })

    const rows = parsed.data;

    const inputRaw = rows.map(row => [
        Number(row.SepalLengthCm),
        Number(row.SepalWidthCm),
        Number(row.PetalLengthCm),
        Number(row.PetalWidthCm),
    ])


    return inputRaw
}

// verileri alalım
const {xsRaw,ysRaw} = readCsv("train.csv")
const inputs = readTestCsv("test.csv")


// normalizasyon işlemi
function normalize(data){
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normalize = dataT.sub(min).div(max.sub(min)) // normalizasyon işlemi

    return {normalize,min,max}
}

// normalize işlemi
const {normalize : xsNorm, min : xsMin, max : xsMax} = normalize(xsRaw)
const {normalize : ysNorm, min : ysMin, max : ysMax} = normalize(ysRaw)

// model oluşturma
async function createModel(){
    const model = tf.sequential()

    model.add(tf.layers.dense({
        units : 16,
        inputShape : [4],
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
        units : 3,
        activation : "softmax",  // softmax ile birden fazla sınıf varsa ona göre dağılım kontrolü yaptırmada kullanıyoruz. BinaryCrossentrpy den farkı birden fazla sınıf kontrol etmesi
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.005
        })
    }))

    model.compile({
        optimizer : tf.train.adam(0.001),
        loss : "categoricalCrossentropy",   // bu da softmax için kullanılıyor. BinaryCrossentropy sigmoid üzerindeyken categoricalCrosssentropy softmax üzerinde
        metrics : ["accuracy","mae"]
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


// model eğitelim ve tahmin yaptıralım
async function trainAndTest(){
    const fit_model = await createModel()

    const lossValues = []
    const maeValues = []
    const accValues = []

    const fit_model_result = await fit_model.fit(xsNorm,ysNorm,{
        epochs : 250,
        validationSplit: 0.2, // yüzde yirmisini validation olarak ayır. yoksa overfitting olabilir
        callbacks : {
            onEpochEnd : async (epoch, logs) => {
                lossValues.push(logs.loss);
                maeValues.push(logs.mae);
                accValues.push(logs.acc || logs.accuracy)
            }
        }
    })

    await fit_model.save("file://./model")

    const loadedModel = await tf.loadLayersModel("file://./model/model.json")
    loadedModel.compile({
        optimizer : tf.train.adam(0.001),
        loss : "categoricalCrossentropy",   // bu da softmax için kullanılıyor. BinaryCrossentropy sigmoid üzerindeyken categoricalCrosssentropy softmax üzerinde
        metrics : ["accuracy","mae"]
    })

    const inputRaw = tf.tensor2d(inputs)
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))
    const predNorm = loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)

    pred.print()

    const [lossTensor,maeTensor,accTensor] = await loadedModel.evaluate(xsNorm,ysNorm)
    console.log("Son Model Kaybı: ",fit_model_result.history.loss.at(-1))
    console.log("Genel Kayıp: ",lossTensor.dataSync()[0])
    console.log("Genel Mutlak Hata Kaybı: ",maeTensor.dataSync()[0])
    console.log("Genel Doğrulama Kaybı: ",accTensor.dataSync()[0])


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

    // veri tahmini ettiriyoruz
    inputRaw.array().then(inputs => {
        pred.array().then(preds => {
            for (let i = 0; i < inputs.length; i++) {
                const maxProb = Math.max(...preds[i]);
                const maxIndex = preds[i].indexOf(maxProb);
                console.log(`${i}. Girdi: [${inputs[i].map(v => v.toFixed(2)).join(", ")}]`);
                console.log(`   Tahmin Olasılıkları: [${preds[i].map(p => p.toFixed(4)).join(", ")}]`);
                console.log(`   Tahmin Edilen Tür: ${speciesList[maxIndex]} (olasılık: ${maxProb.toFixed(4)})`);
            }
        });
    });

    await calculateR2(fit_model, xsNorm, ysNorm, ysMin, ysMax);

}

trainAndTest()
