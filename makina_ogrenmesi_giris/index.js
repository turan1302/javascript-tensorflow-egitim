//  makina öğrenmesi konusuna giris yapılacak

// y = 2x + 1 fonksiyonuna göre öğrenme yapacak model oluşturalım

const tf = require("@tensorflow/tfjs")

const xs = tf.tensor1d([1,2,3,4]) // model eğitmek için örnek girdi verileri
const ys = tf.tensor1d([3,5,7,9]) // 2x+1 formülüne giren girdi verilerinin sonuçları

// bu yukarıdaki iki tensor model eğitmek için kullanılan verilerdir



const model = tf.sequential() // model tanımladık
model.add(tf.layers.dense({  // tek katmak içerecek bir model tanımladık
    units : 1, // 1 nöron kullanılacak. Gelen veriyi bir ağırlık ile çarpıp sapma ekleuecek. 1 değil de başka sayı olsaydı mesela 3 olsaydı o kadar farklı çıktı üretirdi
    inputShape : [1]  // bir sayı girilecek. Mesela a3 gibi 5 gibi 50 gibi sallıyorum
}))

model.compile({
    optimizer : "sgd",  // nöronlar modelin ağırlıklarını vs hesaplıyordu ya. O ağırlık hesaplamada en iyi cevap veren algoritmadır
    loss : "meanSquaredError" // Gerçek değer ile tahmin değer arasındaki farkın karesinin ortalamasını alır. Modelin hatasını buna göre ölçer
})

async function trainAndPredict(){
    // modeli eğitelim
    await model.fit(xs,ys,{  // ilk veri eğitim için girdi verilerimiz, ikinci veri ise eğitim için örnke çıktı verilerimiz
        epochs : 200 // modeli eğitmek için 200 tekrar yapılsın
    });
    console.log("Eğitim tamamlandı")

    // tahmin yaptıralım. 50 girince ne çıkkacak :)
    const sonuc = model.predict(tf.tensor1d([50]))
    sonuc.print()
}

trainAndPredict();
