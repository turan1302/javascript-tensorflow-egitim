/*
Tamam, finansal kredi örneği için eğitim ve test CSV dosyalarında kullanılabilecek 50 satırlık örnek veri oluşturuyorum. Özelliklerimiz şöyle olsun:

yaş (age): 18-65 arası

gelir (income): 1000 - 10000 arası (aylık gelir, TL)

kredi_miktarı (loan_amount): 5000 - 50000 arası (TL)

kredi_süresi (loan_term): 6 - 60 ay

kredi_skora (credit_score): 300 - 850 arası kredi skoru (finansal güvenilirlik göstergesi)

Ve hedef değişkenimiz:

geri_odeme_olasılığı (repayment_probability): 0 ile 1 arası, krediyi zamanında geri ödeme olasılığı.

 */


const tf = require("@tensorflow/tfjs-node")
const {plot} = require("nodeplotlib");
const Papa = require("papaparse")
const fs = require("fs")
const {log} = require("@tensorflow/tfjs-node");

// eğitim verisini içeri dahil edelim
function readTrainCsv(file_directory){
    const file = fs.readFileSync(file_directory,"utf8")

    const parsed = Papa.parse(file,{
        header : true,
        dynamicTyping : true,
        skipEmptyLines:  true
    })

    const rows = parsed.data

    const xsRaw = rows.map(row=>[
        row.age,
        row.income,
        row.loan_amount,
        row.loan_term,
        row.credit_score
    ])

    const ysRaw = rows.map(row=>[
        row.repayment_probability
    ])

    return {xsRaw,ysRaw}
}

// eğitim içeriğini iöçeri aldıralım
function readPredictCsv(file_directory){
    const file = fs.readFileSync(file_directory,"utf8")

    const parsed = Papa.parse(file,{
        header : true,
        dynamicTyping : true,
        skipEmptyLines:  true
    })

    const rows = parsed.data

    const inputRaw = rows.map(row=>[
        row.age,
        row.income,
        row.loan_amount,
        row.loan_term,
        row.credit_score
    ])

    return inputRaw
}

const {xsRaw,ysRaw} = readTrainCsv("veri.csv")
const inputs = readPredictCsv("test.csv")

// normalizasyon işlemi
function normalize(data){
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normalize = dataT.sub(min).div(max.sub(min)) // normalize fonksiyonu

    return {normalize, min, max}
}

// normalize uygulayalım
const {normalize : xsNorm, min : xsMin, max : xsMax} = normalize(xsRaw)
const {normalize : ysNorm, min : ysMin, max : ysMax} = normalize(ysRaw)

// model oluşturalım
 async function createModel(){
    const model = await tf.sequential()

     // model giriş katmanı
     model.add(tf.layers.dense({
         units : 16,
         inputShape : [5],
         activation : "relu",
         kernelRegularizer : tf.regularizers.l2({
             l2 : 0.01
         })
     }))

     // model gizli katman
     model.add(tf.layers.dropout({
         rate : 0.01
     }))

     // model diğer katman
     model.add(tf.layers.dense({
         units : 7,
         activation : "relu",
         kernelRegularizer : tf.regularizers.l2({
             l2 : 0.01
         })
     }))

     // model gizli katman
     model.add(tf.layers.dropout({
         rate : 0.01
     }))

     // model çıkış katmanı
     model.add(tf.layers.dense({
         units : 1,
         activation : "relu",
     }))

     // model derleyelim
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

    const inputRaw = tf.tensor2d(inputs)
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

    // mutlak hata graiği
    plot([{
        x : Array.from({length : maeValues.length},(_,i)=>i+1),
        y : maeValues,
        type : "line",
        name : "MAE",
        line : {
            color : "gray"
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
                    color : "red"
                }
            }],{
                title : "Gerçek - Tahmin Grafiği (ActualPredict)",
                xaxis : {
                    title : "Gerçek"
                },
                yaxis : {
                    title : "Tahmin"
                }
            })
        })
    })

    // histogram grafiği
    pred.data().then(predData=>{
        ysNorm.mul(ysMax.sub(ysMin)).add(ysMin).data().then(actualData=>{

            const error = actualData.map((actual,i)=>actual - predData[i])

            plot([{
                x : Array.from(error),
                type : "histogram",
                name : "histogram",
                line : {
                    color : "green"
                }
            }],{
                title : "Histogra Grafiği (REH)",
                xaxis : {
                    title : "Hata"
                },
                yaxis : {
                    title : "Sıklık"
                }
            })
        })
    })

    // trust analysis
    const [maeTensor,lossTensor] = await loadedModel.evaluate(xsNorm,ysNorm)

    console.log("Son Model Kaybı: ",fit_model_result.history.loss.at(-1))
    console.log("Doğruluk: ",lossTensor.dataSync()[0])

    pred.print()

    // girdi çıktı tahmin
    inputRaw.array().then(inputs=>{
        pred.array().then(preds=>{
            for (let i=0;i<inputs.length; i++){
                console.log(`Girdi: ${inputs[i]} => Tahmin: ${preds[i][0].toFixed(2)} Oran`)
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
