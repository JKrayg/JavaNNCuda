package com.nn.training.metrics;

import org.ejml.simple.SimpleMatrix;

public abstract class Metrics {
    public abstract void getMetrics(SimpleMatrix pred, SimpleMatrix trueVals);
    public abstract double accuracy(SimpleMatrix pred, SimpleMatrix trueVals);
    public abstract double recall(SimpleMatrix pred, SimpleMatrix trueVals);
    public abstract double f1(SimpleMatrix pred, SimpleMatrix trueVals);
}
