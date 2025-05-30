package com.nn.training.callbacks;

public class ExponentialDecay extends LRScheduler {
    private float decayRate;

    public ExponentialDecay(double decayRate) {
        this.decayRate = (float) decayRate;
    }

    public float drop(float learningRate, int epoch) {
        return (float) (learningRate * Math.exp(-decayRate * epoch));
    }
    
}
