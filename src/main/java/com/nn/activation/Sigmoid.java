package com.nn.activation;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.nn.components.Layer;

public class Sigmoid extends ActivationFunction {
    public INDArray execute(INDArray z) {
        int rows = z.rows();
        float[] v = new float[z.rows()];

        for (int i = 0; i < rows; i++) {
            v[i] = (float) (1 / (1 + Math.exp(-(z.getFloat(i)))));
        }
        return Nd4j.create(v);
    }

    public INDArray derivative(INDArray z) {
        // ***
        return z;
    }

    public INDArray gradient(Layer curr, INDArray gradientWrtPreAct) {
        return gradientWrtPreAct.mul(derivative(curr.getPreActivation()));
    }
}
