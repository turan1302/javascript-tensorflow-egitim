const tf = require("@tensorflow/tfjs");

// değeri 7 olan skaler
const scaler = tf.scalar(7)
scaler.print()
console.log("Skalar büyüklük: ",scaler.shape)

/*
çıktı:

Tensor
    7
 */


// [4,5,6] olan vektör çizelim
const vector = tf.tensor1d([4,5,6])
vector.print()
console.log("Vektör büyüklük: ",vector.shape)

/*
çıktı:

Tensor
    [4, 5, 6]
 */

// [1,2,3] ve [4,5,6] ile matris oluşturma işlemini gerçekleştirelim
const matrix = tf.tensor2d([[1,2,3],[4,5,6]])
matrix.print()
console.log("Matrix büyüklük: ",matrix.shape)

/*
çıktı:

Tensor
    [[1, 2, 3],
     [4, 5, 6]]
 */
