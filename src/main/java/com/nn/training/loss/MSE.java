package com.nn.training.loss;


import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import com.nn.components.Layer;

public class MSE extends Loss {
    public float execute(INDArray activations, INDArray pred) {
        System.out.println(Arrays.toString(activations.shape()));
        System.out.println(Arrays.toString(pred.shape()));
        INDArray dif = activations.sub(pred.reshape(pred.shape()[0], 1));
        return dif.mul(dif).sumNumber().floatValue() / pred.length();
    }

    // public INDArray outputGradientWeights(Layer out, Layer prev, float[] preds) {
    //     return new INDArray(0, 0);
    // }

    public INDArray gradient(INDArray activation, INDArray pred) {
        // ***
        System.out.println("&&&&: " + Arrays.toString(activation.shape()));
        System.out.println("!!!!: " + Arrays.toString(pred.shape()));
        INDArray dif = activation.sub(pred.reshape(pred.shape()[0], 1));
        return dif.div(pred.length());
    }
    
}
