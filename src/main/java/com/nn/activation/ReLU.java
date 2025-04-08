package com.nn.activation;

import org.ejml.simple.SimpleMatrix;

import com.nn.components.Layer;

public class ReLU extends ActivationFunction {
    public SimpleMatrix execute(SimpleMatrix z) {
        int rows = z.getNumRows();
        int cols = z.getNumCols();
        SimpleMatrix ez = new SimpleMatrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double curr = z.get(i, j);
                ez.set(i, j, curr > 0 ? curr : 0);
            }
            
        }
        return ez;
    }

    public SimpleMatrix derivative(SimpleMatrix z) {
        int rows = z.getNumRows();
        int cols = z.getNumCols();
        SimpleMatrix dz = new SimpleMatrix(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                dz.set(i, j, z.get(i, j) > 0 ? 1.0 : 0.0);
            }
            
        }
        return dz;
    }

    public SimpleMatrix gradient(Layer prev, SimpleMatrix gradientWrtPreAct) {
        return derivative(prev.getPreActivation()).elementMult(gradientWrtPreAct);
    }
}
