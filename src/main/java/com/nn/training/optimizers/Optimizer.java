package com.nn.training.optimizers;

import org.nd4j.linalg.api.ndarray.INDArray;
import com.nn.components.Layer;
import com.nn.training.normalization.BatchNormalization;
import com.nn.training.normalization.Normalization;

public abstract class Optimizer {
    public abstract INDArray executeWeightsUpdate(Layer l);
    public abstract INDArray executeBiasUpdate(Layer l);
    public abstract INDArray executeShiftUpdate(Normalization b);
    public abstract INDArray executeScaleUpdate(Normalization b);
    public abstract float getLearningRate();
    public abstract void setLearningRate(float learningRate);
    // public abstract float getMomentumDecay();
    // public abstract float getVarianceDecay();
    // public abstract float getEpsilon();
}
