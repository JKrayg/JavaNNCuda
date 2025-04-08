package com.nn.training.loss;

import org.ejml.simple.SimpleMatrix;

import com.nn.components.Layer;

public class BinCrossEntropy extends Loss {
    public double execute(SimpleMatrix activations, SimpleMatrix labels) {
        double sumLoss = 0.0;
        double n = activations.getNumElements();

        for (int i = 0; i < n; i++) {
            double pred = activations.get(i);
            double y = labels.get(i);
            double epsilon = 1e-8;
            // need to prevent log(0)
            pred = Math.max(epsilon, Math.min(1 - epsilon, pred));
            sumLoss += -(y * Math.log(pred) + (1 - y) * Math.log(1 - pred));
        }
        return sumLoss / n;
    }

    public SimpleMatrix gradient(Layer out, SimpleMatrix labels) {
        return out.getActivations().minus(labels);
    }
}
