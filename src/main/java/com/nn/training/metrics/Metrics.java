package com.nn.training.metrics;

import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class Metrics {
    public abstract void getMetrics(INDArray pred, INDArray trueVals);
    public abstract float accuracy(INDArray pred, INDArray trueVals);
    // public abstract float recall(INDArray pred, INDArray trueVals);
    // public abstract float f1(INDArray pred, INDArray trueVals);
}
