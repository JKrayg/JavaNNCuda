package com.nn.training.loss;

import org.ejml.simple.SimpleMatrix;
import com.nn.components.Layer;

public class SparseCrossEntropy extends Loss {
    public double execute(SimpleMatrix activations, SimpleMatrix labels) {
        return 0.0;
    }

    public SimpleMatrix gradient(Layer out, SimpleMatrix labels) {
        // ***
        return out.getActivations().minus(labels);
    }
}
