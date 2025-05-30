package com.nn.training.callbacks;

public class StepDecay extends LRScheduler {
    private float dropRate;
    private int freq;
    private float learningRate;

    public StepDecay(double learningRate, double dropRate, int freq) {
        this.learningRate = (float) learningRate;
        this.dropRate = (float)dropRate;
        this.freq = freq;

    }

    public float drop(float lr, int epoch) {
        if (epoch % freq == 0) {
            this.learningRate = (float) (lr * Math.pow(dropRate, Math.floorDiv(epoch, freq)));
        }

        return this.learningRate;
        
    }
}
