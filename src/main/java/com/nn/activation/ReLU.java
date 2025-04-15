package com.nn.activation;


import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import com.nn.components.Layer;

public class ReLU extends ActivationFunction {
    public INDArray execute(INDArray z) {
        INDArray az = z.gt(0).castTo(DataType.FLOAT).mul(z);
        return az;
    }

    public INDArray derivative(INDArray z) {
        INDArray dz = z.gt(0).castTo(DataType.FLOAT);
        return dz;
    }

    public INDArray gradient(Layer prev, INDArray gradientWrtPreAct) {
        return derivative(prev.getPreActivation()).mul(gradientWrtPreAct);
    }
}
