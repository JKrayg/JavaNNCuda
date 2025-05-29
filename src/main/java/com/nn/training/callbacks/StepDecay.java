package com.nn.training.callbacks;

public class StepDecay extends LRScheduler {
    private float dropRate;
    private int freq;

    public StepDecay(double dropRate, int freq) {
        this.dropRate = (float)dropRate;
        this.freq = freq;
    }

    public float execute(float lr, int epoch) {
        return (float) (lr * Math.pow(dropRate, Math.floor(epoch / freq)));
    }
}
