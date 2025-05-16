package com.nn.training.loss;


import org.nd4j.linalg.api.ndarray.INDArray;
import com.nn.components.Layer;

public class MSE extends Loss {
    public float execute(INDArray activations, INDArray labels) {
        INDArray dif = activations.sub(labels);
        return dif.mul(dif).sumNumber().floatValue() / labels.length();
    }

    // public INDArray outputGradientWeights(Layer out, Layer prev, float[] labels) {
    //     return new INDArray(0, 0);
    // }

    public INDArray gradient(Layer out, INDArray labels) {
        // ***
        INDArray dif = out.getActivations().sub(labels);
        return dif.div(labels.length());
    }
    
}
