package com.nn.training.regularizers;

import org.nd4j.linalg.api.ndarray.INDArray;

public class L2 extends Regularizer {
    private float lambda;

    public L2() {
        this.lambda = 0.01f;
    }

    public L2(double lam) {
        this.lambda = (float) lam;
    }

    public INDArray regularize(INDArray weights) {
        INDArray w = weights.mul(lambda);
        return w;
    }
    
}
