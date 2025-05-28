const tf = require("@tensorflow/tfjs-node");  // tfjs nin node kısmı. Dosya laydetme işlemlerinde yardımcı olur

const xs = tf.tensor1d([1, 2, 3, 4]);
const ys = tf.tensor1d([3, 5, 7, 9]); // y = 2x+1

const model = tf.sequential();
model.add(tf.layers.dense({
    units: 1,
    inputShape: [1]
}));

model.compile({
    optimizer: "sgd",
    loss: "meanSquaredError"
});

async function trainAndPredict() {
    await model.fit(xs, ys, {
        epochs: 500
    });

    // MODELİ KAYDET
    await model.save('file://./model');
    console.log("Model Kaydedildi :))");

    // MODELİ YÜKLE
    const loadedModel = await tf.loadLayersModel("file://./model/model.json");

    // YENİ GİRİŞ VERİSİYLE TAHMİN YAP
    const sonuc = loadedModel.predict(tf.tensor1d([5]));
    sonuc.print();
}

trainAndPredict();
