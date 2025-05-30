package com.nn.training.callbacks;

public abstract class LRScheduler extends Callback{
    public abstract float drop(float learningRate, int epoch);
}
