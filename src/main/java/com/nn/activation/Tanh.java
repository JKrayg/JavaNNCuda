package com.nn.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.nn.components.Layer;

public class Tanh extends ActivationFunction {
    public INDArray execute(INDArray z) {
        int rows = z.rows();
        float[] v = new float[rows];

        for (int i = 0; i < rows; i++) {
            float curr = z.getFloat(i);
            v[i] = (float) ((Math.exp(curr) - Math.exp(-curr)) / (Math.exp(curr) + Math.exp(-curr)));
        }
        return Nd4j.create(v);
    }

    public INDArray derivative(INDArray z) {
        // ***
        return z;
    }

    public INDArray gradient(INDArray preactivation, INDArray gradientWrtPreAct) {
        return gradientWrtPreAct.mmul(derivative(preactivation));
    }
}
