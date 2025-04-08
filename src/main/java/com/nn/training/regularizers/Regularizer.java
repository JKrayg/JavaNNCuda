package com.nn.training.regularizers;

import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class Regularizer {
    public abstract INDArray regularize(INDArray weights);
}
