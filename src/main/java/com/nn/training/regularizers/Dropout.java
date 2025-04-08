package com.nn.training.regularizers;

import org.ejml.simple.SimpleMatrix;
import java.util.Random;

public class Dropout extends Regularizer{
    double p;

    public Dropout(double p) {
        this.p = p;
    }

    public SimpleMatrix regularize(SimpleMatrix activations) {
        int cols = activations.getNumCols();
        int rows = activations.getNumRows();
        SimpleMatrix drop = activations.copy();
        Random rand = new Random();

        for (int i = 0; i < cols; i++) {
            if (rand.nextDouble() < p) {
                drop.setColumn(i, new SimpleMatrix(rows, 1));
            }
        }
        return drop;
    }
    
}
