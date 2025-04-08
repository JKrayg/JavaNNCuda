package com.nn.activation;

import org.ejml.simple.SimpleMatrix;

import com.nn.components.Layer;

public class Sigmoid extends ActivationFunction {
    public SimpleMatrix execute(SimpleMatrix z) {
        int rows = z.getNumRows();
        double[] v = new double[z.getNumRows()];

        for (int i = 0; i < rows; i++) {
            v[i] = 1 / (1 + Math.exp(-(z.get(i))));
        }
        return new SimpleMatrix(v);
    }

    public SimpleMatrix derivative(SimpleMatrix z) {
        // ***
        return z;
    }

    public SimpleMatrix gradient(Layer curr, SimpleMatrix gradientWrtPreAct) {
        return gradientWrtPreAct.elementMult(derivative(curr.getPreActivation()));
    }
}
