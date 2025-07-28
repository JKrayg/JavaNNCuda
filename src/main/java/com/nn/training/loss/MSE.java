package com.nn.training.loss;


import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import com.nn.components.Layer;

public class MSE extends Loss {
    public float execute(INDArray activations, INDArray pred) {
        // System.out.println(Arrays.toString(activations.shape()));
        // System.out.println(Arrays.toString(pred.shape()));
        INDArray dif = activations.sub(pred.reshape(pred.shape()[0], 1));
        return dif.mul(dif).sumNumber().floatValue() / pred.length();
    }

    // public INDArray outputGradientWeights(Layer out, Layer prev, float[] preds) {
    //     return new INDArray(0, 0);
    // }

    public INDArray gradient(Layer out, INDArray pred) {
        // ***
        // System.out.println("&&&&: " + Arrays.toString(out.getActivations().shape()));
        // System.out.println("!!!!: " + Arrays.toString(pred.shape()));
        INDArray dif = out.getActivations().sub(pred.reshape(pred.shape()[0], 1));
        // System.out.println("3434: " + Arrays.toString(dif.div(pred.length()).shape()));
        return dif.div(pred.length());
    }

    // @Override
    // public INDArray gradient(Layer out, INDArray preds) {
    //     // TODO Auto-generated method stub
    //     throw new UnsupportedOperationException("Unimplemented method 'gradient'");
    // }

    
}
