// broadcasting ile küçük tensorün büyük tensorle uyumlu hale getirilerek işlem yapılmasıdır

/*
    ÖRNEK

    [[1, 2, 3],
 [4, 5, 6]]   +   [10, 20, 30]  → Broadcasting uygulanır

 BURADAKİ + İŞLEMİ OLAYI 'add' fonksiyonu kullanılarak yapılmaktadır

    SONUÇ

    [[11, 22, 33],
 [14, 25, 36]]
 */



const tf = require("@tensorflow/tfjs")
const {tensor1d} = require("@tensorflow/tfjs");

/*
AŞAĞIDAKİ İŞLEME BROADCASTING UYGULA

A = [[1, 2, 3],
     [4, 5, 6]]

B = [10, 20, 30]
 */

const t1 = tf.tensor2d([[1,2,3],[4,5,6]])
const t2 = tensor1d([10,20,30])

const sonuc = t1.add(t2)  // add fonksiyony ile broadcasting islemi ayarlamış olduk
sonuc.print()

/*
    ekran çıktısı:

Tensor
    [[11, 22, 33],
     [14, 25, 36]]
 */
