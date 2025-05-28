// tensor sekli degistirmek icin reshape kullanırız

// ilk parametre satırdır
// ikinci parametre sütundur

const tf = require("@tensorflow/tfjs");

/*
Aşağıdaki vektörü oluştur ve 2x2 matrise dönüştür:

[10, 20, 30, 40]
 */

const v1 = tf.tensor1d([10,20,30,40])
const m1 = v1.reshape([2,2])  // ilm parametre satır, ,kinci parametre sütun

m1.print()
console.log("Boyut: ",m1.shape)

/*
ekran çıktısı:

Tensor
    [[10, 20],
     [30, 40]]
Boyut:  [ 2, 2 ]
 */
