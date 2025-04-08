package com.nn.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import com.nn.components.Layer;

public abstract class ActivationFunction {
    public abstract INDArray execute(INDArray z);
    public abstract INDArray derivative(INDArray z);
    public abstract INDArray gradient(Layer curr, INDArray gradientWrtPreAct);
}
