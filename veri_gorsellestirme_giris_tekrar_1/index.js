/*
🎯 Konu: y = 2x + 1 fonksiyonunu öğrenen bir model
Ve bu modelin kayıp (loss) değerini her epoch’ta nasıl düşürdüğünü grafikle göstereceğiz.
 */

const tf = require("@tensorflow/tfjs-node")
const {plot} = require("nodeplotlib");

// x ve y verileri: y = 2x + 1
const xs = tf.tensor1d([0, 1, 2, 3, 4])
const ys = tf.tensor1d([1, 3, 5, 7, 9]) // 2x + 1

async function trainAndPredict() {
    const model = tf.sequential()
    model.add(tf.layers.dense({
        units: 1,
        inputShape: [1]
    }))
    model.compile({
        optimizer: "adam",
        loss: "meanSquaredError"
    })

    const lossValues = []

    await model.fit(xs, ys, {
        epochs: 100,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                lossValues.push(logs.loss)
            }
        }
    })

    // Grafik verisi
    const data = [{
        x: Array.from({length: lossValues.length}, (_, i) => i + 1),
        y: lossValues,
        type: 'line',
        name: 'Loss'
    }];

    // Grafik ayarları
    const layout = {
        title: 'Eğitim Kaybı Grafiği',
        xaxis: { title: 'Epoch' },
        yaxis: { title: 'Loss Değeri' }
    };

    plot(data,layout);
}

trainAndPredict()
