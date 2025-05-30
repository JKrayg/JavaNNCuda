package com.nn.training.optimizers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.nn.components.Layer;
import com.nn.layers.Dense;
import com.nn.training.normalization.Normalization;

public class RMSProp extends Optimizer {
    private float learningRate;

    public RMSProp(float learningRate) {
        this.learningRate = learningRate;
    }

    public INDArray executeWeightsUpdate(Layer l) {
        // **
        return ((Dense)l).getWeights();
    }

    public INDArray executeBiasUpdate(Layer l) {
        // **
        return l.getBias();
    }

    public INDArray executeShiftUpdate(Normalization n) {
        return Nd4j.create(0, 0);
    }

    public INDArray executeScaleUpdate(Normalization n) {
        return Nd4j.create(0, 0);
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public float getLearningRate() {
        return learningRate;
    }
}
