// iki vektörü toplama yapacakken aynı uzunluklarda, aynı boyutlarda olmaları gereklidir

const tf = require("@tensorflow/tfjs")

const t1 = tf.tensor1d([3,5,7])
const t2 = tf.tensor1d([1,2,3]);

const toplam = t1.add(t2);
toplam.print()
console.log("Boyut: ",toplam.shape)

/*
    çıktı:

    Tensor
    [4, 7, 10]
Boyut:  [ 3 ]

 */
