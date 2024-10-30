# BLAS usage examples

The documents and presentations of the tutorial can be found in the `doc` directory.

## Collection of matrix examples

The [Matrix Market](https://math.nist.gov/MatrixMarket/index.html) repository of test data for use in performance analysis of algorithms for numerical linear algebra. It contains multiple collections of test data, such as the [Harwell-Boeing](https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/) and [SPARSKIT](https://math.nist.gov/MatrixMarket/data/SPARSKIT/) collections. The data are provided in [multiple text file formats](https://math.nist.gov/MatrixMarket/formats.html), including [Harwell-Boeing Exchange Format](https://math.nist.gov/MatrixMarket/formats.html#hb) and the [Matrix Market Exchange Format](https://math.nist.gov/MatrixMarket/formats.html#MMformat). We are using the Matrix Market Exchange Format due to its simplicity. More information regarding the format can be found in its [official documentation](https://math.nist.gov/MatrixMarket/reports/MMformat.ps).

In this example the matrices user are:
- [`Harwell-Boeing/bcsstruc1/bcsstk01`](https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/bcsstruc1/bcsstk01.html): BCS Structural Engineering Matrices (eigenvalue matrices)
    - Type: real symmetric positive definite
    - Size: 48 x 48, 224 entries

