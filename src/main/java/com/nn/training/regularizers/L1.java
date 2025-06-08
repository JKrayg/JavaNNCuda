package com.nn.training.regularizers;

import org.nd4j.linalg.api.ndarray.INDArray;

public class L1 extends Regularizer{
    private float lambda;

    public L1() {
        this.lambda = 0.01f;
    }

    public L1(float lam) {
        this.lambda = lam;
    }

    public INDArray regularize(INDArray weights) {
        INDArray r = weights.gt(0).castTo(weights.dataType());
        INDArray a = weights.lt(0).castTo(weights.dataType());
        // INDArray w = weights.dup();
        // int rows = w.rows();
        // int cols = w.columns();
        // for (int i = 0; i < rows; i++) {
        //     for (int j = 0; j < cols; j++) {
        //         float currWeight = w.getFloat(i, j);
        //         w.put(i, j, currWeight > 0 ? 1 : (currWeight < 0 ? -1 : 0));
        //     }
        // }
        return r.sub(a).mul(lambda);
    }
}
