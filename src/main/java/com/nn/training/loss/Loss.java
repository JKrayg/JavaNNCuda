package com.nn.training.loss;

import org.ejml.simple.SimpleMatrix;

import com.nn.components.Layer;

public abstract class Loss {
    public abstract double execute(SimpleMatrix activations, SimpleMatrix labels);

    // public abstract SimpleMatrix outputGradientWeights(Layer out, Layer prev, double[] labels);

    public abstract SimpleMatrix gradient(Layer out, SimpleMatrix labels);
    
}
