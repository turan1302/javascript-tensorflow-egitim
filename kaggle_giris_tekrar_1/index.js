/*
burada kişinin sigorta ücretini tahmin edeceğiz
 */

const tf = require("@tensorflow/tfjs-node")
const {plot} = require("nodeplotlib");
const Papa = require("papaparse")
const fs = require("fs")
const {train} = require("@tensorflow/tfjs-node");

// eğitim verisi okuma işlemi
function readTrainCsv(file_directory) {
    const file = fs.readFileSync(file_directory, "utf8")

    const parsed = Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true
    })

    const rows = parsed.data

    const xsRaw = rows.map(row=>[
        row.age,
        row.sex==="male" ? 1 : 0,
        row.bmi,
        row.children,
        row.smoker==="yes" ? 1 : 0,
    ])

    const ysRaw = rows.map(row=>[
        row.charges
    ])

    return {xsRaw,ysRaw}
}

//test verisi okuma islemi
function readTestCsv(file_directory) {
    const file = fs.readFileSync(file_directory, "utf8")

    const parsed = Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true
    })

    const rows = parsed.data

    const inputRaw = rows.map(row=>[
        row.age,
        row.sex==="male" ? 1 : 0,
        row.bmi,
        row.children,
        row.smoker==="yes" ? 1 : 0,
    ])


    return inputRaw
}

// verileri alalım
const {xsRaw,ysRaw} = readTrainCsv("train.csv")
const inputs = readTestCsv("test.csv")

// normalizasyon fonksiyonu
function normalize(data){
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normalize = dataT.sub(min).div(max.sub(min)) // normalize işlemi

    return {normalize,min,max}
}

// normalize işlemi
const {normalize : xsNorm, min : xsMin, max : xsMax} = normalize(xsRaw)
const {normalize : ysNorm, min : ysMin, max : ysMax} = normalize(ysRaw)

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

// model eğitme ve tahmin kısmı
async function trainAndPredict(){

}

trainAndPredict()
