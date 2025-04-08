package com.nn.layers;

import org.ejml.simple.SimpleMatrix;
import com.nn.activation.ActivationFunction;
import com.nn.components.*;
import com.nn.training.regularizers.Regularizer;

// OutputLayer and Dense are pretty much the same class
public class Dense extends Layer {
    public Dense(int numNeurons, ActivationFunction actFunc) {
        super(numNeurons, actFunc);
    }

    public Dense(int numNeurons, ActivationFunction actFunc, int inputSize) {
        super(numNeurons, actFunc, inputSize);
    }
}