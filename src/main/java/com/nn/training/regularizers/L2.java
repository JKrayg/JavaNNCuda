package com.nn.training.regularizers;

import org.ejml.simple.SimpleMatrix;

public class L2 extends Regularizer {
    private double lambda;

    public L2() {
        this.lambda = 0.01;
    }

    public L2(double lam) {
        this.lambda = lam;
    }

    public SimpleMatrix regularize(SimpleMatrix weights) {
        SimpleMatrix w = weights.scale(lambda);
        return w;
    }
    
}
