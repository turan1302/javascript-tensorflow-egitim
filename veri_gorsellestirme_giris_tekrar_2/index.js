/*
Soru:
Bir öğrencinin sınavda aldığı puanı tahmin eden basit bir yapay sinir ağı modeli yapacaksın. Girdi olarak:

Ders çalışma saati

Ders dışı ekstra tekrar saati

Uykusunun kaç saat olduğu

çıktı olarak ise sınavdan alacağı tahmini puanı (0-100 arasında) verecek.

Yapacakların:

Küçük bir veri seti oluştur (5-10 tane örnek).

Girdi verilerini ve çıkış verilerini normalize et.

TensorFlow ile basit bir sequential model oluştur (2 katmanlı, uygun aktivasyonlarla).

Modeli eğit (örneğin 200-300 epoch).

Eğitim sırasında her epoch’un kayıp değerlerini bir diziye kaydet.

Eğitimi bitince kayıp değerlerini nodeplotlib ile grafiğe döküp göster.

Eğitilen modeli kullanarak yeni bir veri ile tahmin yap ve sonucu console’a yazdır.
 */

const tf = require("@tensorflow/tfjs-node")
const {plot} = require("nodeplotlib");

const xsRaw = [
    [2, 3, 6],  // 2 saat çalıştı, 3 saat tekrar yaptı, 6 saat uyudu
    [4, 1, 7],
    [1, 0, 5],
    [5, 4, 8],
    [3, 2, 6],
    [6, 3, 7],
    [2, 1, 5],
    [7, 5, 8],
    [4, 3, 7],
    [1, 0, 4]
]

const ysRaw = [
    [70],  // 70 puan aldı
    [80],
    [50],
    [90],
    [75],
    [95],
    [65],
    [100],
    [85],
    [45]
]

// normalizasyon fonksiyonunu yazalım
function normalize(data){
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normalize = dataT.sub(min).div(max.sub(min)) // bormalize uygulayalım
    return {normalize, min, max}
}

// normalizasyon uygulayalım
const {normalize : xsNorm, min : xsMin, max : xsMax} = normalize(xsRaw)
const {normalize : ysNorm, min : ysMin, max : ysMax} = normalize(ysRaw)

// model oluşturma fonkiyonu
async function createModel(){
    const model = tf.sequential()
    model.add(tf.layers.dense({
        units : 16,
        inputShape : [3],
        activation : "relu",
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.01
        })
    }))

    model.add(tf.layers.dropout({
        rate : 0.3
    }))

    model.add(tf.layers.dense({
        units : 8,
        activation : "relu",
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.01
        })
    }))

    model.add(tf.layers.dropout({
        rate : 0.3
    }))

    model.add(tf.layers.dense({
        units : 4,
        activation : "relu",
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.01
        })
    }))

    model.add(tf.layers.dropout({
        rate : 0.3
    }))

    model.add(tf.layers.dense({
        units : 2,
        activation : "relu",
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.01
        })
    }))

    model.add(tf.layers.dropout({
        rate : 0.3
    }))

    model.add(tf.layers.dense({
        units : 1,
        activation : "linear",
    }))

    // model derleyelim
    model.compile({
        optimizer : "adam",
        loss : "meanSquaredError"
    })

    return model
}

async function trainAndPredict(){
    const fit_model = await createModel()

    const lossValues = []

    await fit_model.fit(xsNorm,ysNorm,{
        epochs : 250,
        callbacks : {
            onEpochEnd : async (epoch, logs) => {
                lossValues.push(logs.loss)
            }
        }
    })

    plot([{
        x : Array.from({length: lossValues.length}, (_, i) => i + 1),
        y : lossValues,
        type : "line",
        name : "Loss"
    }],{
        title : "Öğrenme Kaybı",
        yaxis : {
            title : "Kayıp"
        },
        xaxis : {
            title : "Epochs"
        }
    })

    await fit_model.save("file://./model")

    const loadedModel = await tf.loadLayersModel("file://./model/model.json")

    const inputRaw = tf.tensor2d([[3, 2, 6]])
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))

    const predNorm = loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)



    pred.print()
}

trainAndPredict()
