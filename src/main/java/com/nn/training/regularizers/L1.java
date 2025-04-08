package com.nn.training.regularizers;

import org.ejml.simple.SimpleMatrix;

public class L1 extends Regularizer{
    private double lambda;

    public L1() {
        this.lambda = 0.01;
    }

    public L1(double lam) {
        this.lambda = lam;
    }

    public SimpleMatrix regularize(SimpleMatrix weights) {
        SimpleMatrix w = weights.copy();
        int rows = w.getNumRows();
        int cols = w.getNumCols();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double currWeight = w.get(i, j);
                w.set(i, j, currWeight > 0 ? 1 : (currWeight < 0 ? -1 : 0));
            }
        }
        return w.scale(lambda);
    }
}
