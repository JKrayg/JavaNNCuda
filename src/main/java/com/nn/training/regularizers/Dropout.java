package com.nn.training.regularizers;

import java.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Dropout extends Regularizer{
    float p;

    public Dropout(float p) {
        this.p = p;
    }

    public INDArray regularize(INDArray activations) {
        int cols = activations.columns();
        int rows = activations.rows();
        INDArray drop = activations.dup();
        Random rand = new Random();

        for (int i = 0; i < cols; i++) {
            if (rand.nextFloat() < p) {
                drop.putColumn(i, Nd4j.create(rows, 1));
            }
        }
        return drop;
    }
    
}
