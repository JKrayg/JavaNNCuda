package com.nn.activation;

import org.ejml.simple.SimpleMatrix;
import com.nn.components.Layer;

public abstract class ActivationFunction {
    public abstract SimpleMatrix execute(SimpleMatrix z);
    public abstract SimpleMatrix derivative(SimpleMatrix z);
    public abstract SimpleMatrix gradient(Layer curr, SimpleMatrix gradientWrtPreAct);
}
