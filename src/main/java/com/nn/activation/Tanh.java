package com.nn.activation;

import org.ejml.simple.SimpleMatrix;

import com.nn.components.Layer;

public class Tanh extends ActivationFunction {
    public SimpleMatrix execute(SimpleMatrix z) {
        double[] v = new double[z.getNumRows()];

        for (int i = 0; i < z.getNumRows(); i++) {
            double curr = z.get(i);
            v[i] = (Math.exp(curr) - Math.exp(-curr)) / (Math.exp(curr) + Math.exp(-curr));
        }
        return new SimpleMatrix(v);
    }

    public SimpleMatrix derivative(SimpleMatrix z) {
        // ***
        return z;
    }

    public SimpleMatrix gradient(Layer curr, SimpleMatrix gradientWrtPreAct) {
        return gradientWrtPreAct.mult(derivative(curr.getPreActivation()));
    }
}
