package com.nn.training.loss;


import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import com.nn.components.Layer;

public class MSE extends Loss {
    int count = 0;
    public float execute(INDArray activations, INDArray labels) {
        // System.out.println("---: " + Arrays.toString(activations.shape()));
        // System.out.println("^^^: " + Arrays.toString(labels.shape()));
        INDArray dif = activations.sub(labels);
        count += 1;
        // System.out.println(count);
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
