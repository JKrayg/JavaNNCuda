package com.nn.training.regularizers;

import org.ejml.simple.SimpleMatrix;

public abstract class Regularizer {
    public abstract SimpleMatrix regularize(SimpleMatrix weights);
}
