package com.nn.activation;

import org.nd4j.linalg.api.ndarray.INDArray;

import com.nn.components.Layer;

public class Linear extends ActivationFunction {
    public INDArray execute(INDArray z) {
        return z;
    }

    public INDArray derivative(INDArray z) {
        // ***
        return z;
    }

    public INDArray gradient(INDArray preactivations, INDArray gradientWrtPreAct) {
        return gradientWrtPreAct;
    }
}
