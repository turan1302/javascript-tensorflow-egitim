const tf = require("@tensorflow/tfjs");

// scalar
const scalar = tf.scalar(5);
scalar.print();

/*
    çıktı:

Tensor
    5

 */

// vektorel
const vector = tf.tensor1d([1,2,3]);
vector.print()

/*

çıktı

Tensor
    [1, 2, 3]
 */

// matris kısmı
const matrix = tf.tensor2d([[1,2],[3,4]])
matrix.print()

/*
    çıktı

Tensor
    [[1, 2],
     [3, 4]]
 */
