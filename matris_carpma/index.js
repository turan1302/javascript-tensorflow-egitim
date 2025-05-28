// matris çarpma işleminde ilk matrisin sütun sayısı, ikinci matrisin satır sayısına eşit olacak

const tf = require("@tensorflow/tfjs")

const m1 = tf.tensor2d([[1,2,3],[4,5,6]]) // 2 satır 3 sütun
const m2 = tf.tensor2d([[7,8],[9,10],[11,12]]) // 3 satır, 2 sütun

const carpim = m1.matMul(m2);
carpim.print();
console.log("Boyut: ",carpim.shape)

/*
Çıktı:

Tensor
    [[58 , 64 ],
     [139, 154]]
Boyut:  [ 2, 2 ]
 */
