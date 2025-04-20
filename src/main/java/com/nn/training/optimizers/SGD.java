package com.nn.training.optimizers;

import org.nd4j.linalg.api.ndarray.INDArray;
import com.nn.components.Layer;
import com.nn.layers.Dense;
import com.nn.training.normalization.BatchNormalization;
import com.nn.training.normalization.Normalization;

public class SGD extends Optimizer {
    private float learningRate;

    public SGD(double learningRate) {
        this.learningRate = (float) learningRate;
    }

    public INDArray executeWeightsUpdate(Layer l) {
        return ((Dense)l).getWeights().sub(((Dense)l).getGradientWeights().mul(learningRate));
    }

    public INDArray executeBiasUpdate(Layer l) {
        return l.getBias().sub(l.getGradientBias().mul(learningRate));
    }

    public INDArray executeShiftUpdate(Normalization n) {
        return n.getShift().sub(n.getGradientShift().mul(learningRate));
    }

    public INDArray executeScaleUpdate(Normalization n) {
        return n.getScale().sub(n.getGradientScale().mul(learningRate));
    }

    public float getLearningRate() {
        return learningRate;
    }
}
