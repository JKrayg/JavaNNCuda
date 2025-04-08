package com.nn.layers;

import com.nn.activation.ActivationFunction;
import com.nn.components.*;

// OutputLayer and Dense are pretty much the same class
public class Dense extends Layer {
    public Dense(int numNeurons, ActivationFunction actFunc) {
        super(numNeurons, actFunc);
    }

    public Dense(int numNeurons, ActivationFunction actFunc, int inputSize) {
        super(numNeurons, actFunc, inputSize);
    }
}