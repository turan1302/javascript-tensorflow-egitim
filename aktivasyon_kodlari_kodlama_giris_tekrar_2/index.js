/*
📊 Yeni Örnek Fikri: Ev Fiyat Tahmini
🎯 Amaç:
Aşağıdaki verilere göre bir evin tahmini fiyatını (bin $ olarak) bulalım:

Özellik	Açıklama
Oda sayısı	Kaç oda var (örneğin: 2, 3, 4...)
Metrekare	Evin büyüklüğü (örneğin: 70, 90..)
Semt puanı	0–1 arasında (mahalle kalitesi)
Balkon var mı?	1 = Var, 0 = Yok
 */

const xsRaw = [
    [2, 70, 0.6, 1],
    [3, 90, 0.8, 1],
    [1, 45, 0.5, 0],
    [4, 120, 0.9, 1],
    [2, 60, 0.4, 0],
    [3, 85, 0.7, 1],
    [1, 40, 0.3, 0],
    [5, 150, 1.0, 1],
    [3, 95, 0.85, 1],
    [2, 75, 0.6, 0]
]

const ysRaw = [
    [150],
    [220],
    [95],
    [300],
    [120],
    [210],
    [85],
    [350],
    [240],
    [160]
]

// kütüphaneyi kurarım
const tf = require("@tensorflow/tfjs-node")

// normalizasyon fonksiyonu oluşturalım
function normalize(data){
    const dataT = tf.tensor2d(data)
    const min = dataT.min(0) // axis boyunca min
    const max = dataT.max(0) // axis boyunca max
    const normalize = dataT.sub(min).div(max.sub(min)) // normalizasyon fonksiyonu
    return {normalize,min,max}
}

// normalizasyon uygulayalım
const {normalize : xsNorm, min : xsMin, max : xsMax} = normalize(xsRaw)
const {normalize : ysNorm, min : ysMin, max : ysMax} = normalize(ysRaw)

// model oluşturma fonksiyonu
async function createModel(activation){
    const model = await tf.sequential()

    // model ilkm katman
    model.add(tf.layers.dense({
        units : 16, // 16 nöron
        inputShape : [4], // 4 giriş
        activation  : activation,
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.01
        })
    }))

    // drop ekleyeklim
    model.add(tf.layers.dropout({
        rate : 0.3
    }))

    // model ikinci katman
    model.add(tf.layers.dense({
        units : 8, // 8 nöron
        activation  : activation,
        kernelRegularizer : tf.regularizers.l2({
            l2 : 0.01
        })
    }))

    // drop ekleyeklim
    model.add(tf.layers.dropout({
        rate : 0.3
    }))

    // çıkış katmanı
    model.add(tf.layers.dense({
        units : 1, // 8 nöron
        activation  : "linear",
    }))

    // modeli derleyelim
    model.compile({
        optimizer : "adam",
        loss : "meanSquaredError"
    })

    return model
}

// model eğitelim ve tahmöin yaptıralım
async function trainAndPredict(activation){
    const model = await createModel(activation)

    const fit_model = await model.fit(xsNorm,ysNorm,{
        epochs : 500
    })

    // model kaydedelim
    await model.save(`file://./model-${activation}`)

    // modeli çağıralım
    const loadedModel = await tf.loadLayersModel(`file://./model-${activation}/model.json`)

    const inputRaw = tf.tensor2d([[3, 100, 0.75, 1]])
    const inputNorm = inputRaw.sub(xsMin).div(xsMax.sub(xsMin))

    const predNorm = await loadedModel.predict(inputNorm)
    const pred = predNorm.mul(ysMax.sub(ysMin)).add(ysMin)

    // doğruluk analizi
    const trust_analysis = await model.evaluate(xsNorm,ysNorm)

    console.log(`------------${activation.toUpperCase()} ALGORITMASI`)
    console.log("Son Kayıp: ",fit_model.history.loss.at(-1))
    console.log("Doğruluk: ",trust_analysis.dataSync()[0])
    pred.print()
}

function run(){
     trainAndPredict("relu")
     trainAndPredict("sigmoid")
     trainAndPredict("tanh")
}
run()
