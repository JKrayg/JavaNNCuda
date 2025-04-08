package com.nn.training.optimizers;


import org.ejml.simple.SimpleMatrix;
import com.nn.components.Layer;
import com.nn.training.normalization.BatchNormalization;
import com.nn.training.normalization.Normalization;

public abstract class Optimizer {
    public abstract SimpleMatrix executeWeightsUpdate(Layer l);
    public abstract SimpleMatrix executeBiasUpdate(Layer l);
    public abstract SimpleMatrix executeShiftUpdate(Normalization b);
    public abstract SimpleMatrix executeScaleUpdate(Normalization b);
    public abstract double getLearningRate();
    // public abstract double getMomentumDecay();
    // public abstract double getVarianceDecay();
    // public abstract double getEpsilon();
}
