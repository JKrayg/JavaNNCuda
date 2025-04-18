package com.nn.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import com.nn.components.Layer;

public class MathUtils {
    // weighted sum ∑(wi⋅xi)+b
    public INDArray weightedSum(Layer prevLayer, Layer currLayer) {
        return getWeightedSum(prevLayer.getActivations(), currLayer);
    }

    public INDArray weightedSum(INDArray inputData, Layer currLayer) {
        return getWeightedSum(inputData, currLayer);
    }

    private static INDArray getWeightedSum(INDArray prev, Layer curr) {
        INDArray weights = curr.getWeights();
        INDArray biasT = curr.getBias().transpose();
        INDArray weighted  = prev.mmul(weights).add(biasT);

        return weighted;
    }

    public float std(INDArray v) {
        int numElements = (int) v.length();
        float mean = (v.sumNumber().floatValue() / numElements);
        float s = 0;

        for (int i = 0; i < numElements; i++) {
            s += (v.getFloat(i) - mean) * (v.getFloat(i) - mean);
        }

        return (float) Math.sqrt(s / numElements);
    }
}
