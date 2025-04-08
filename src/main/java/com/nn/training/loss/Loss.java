package com.nn.training.loss;

import org.nd4j.linalg.api.ndarray.INDArray;
import com.nn.components.Layer;

public abstract class Loss {
    public abstract float execute(INDArray activations, INDArray labels);

    // public abstract INDArray outputGradientWeights(Layer out, Layer prev, float[] labels);

    public abstract INDArray gradient(Layer out, INDArray labels);
    
}
