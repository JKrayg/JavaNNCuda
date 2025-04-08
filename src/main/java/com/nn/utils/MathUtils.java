package com.nn.utils;

import org.ejml.simple.SimpleMatrix;
import com.nn.components.Layer;

public class MathUtils {
    // weighted sum ∑(wi⋅xi)+b
    public SimpleMatrix weightedSum(Layer prevLayer, Layer currLayer) {
        return getWeightedSum(prevLayer.getActivations(), currLayer);
    }

    public SimpleMatrix weightedSum(SimpleMatrix inputData, Layer currLayer) {
        return getWeightedSum(inputData, currLayer);
    }

    private static SimpleMatrix getWeightedSum(SimpleMatrix prev, Layer curr) {
        // System.out.println("<<<<<<<<<<<<<<<<<");
        // System.out.println(prev.getNumRows() + "x" + prev.getNumCols());
        // System.out.println(curr.getWeights().getNumRows() + "x" + curr.getWeights().getNumCols());
        // System.out.println("<<<<<<<<<<<<<<<<<<");
        SimpleMatrix weights = curr.getWeights();
        SimpleMatrix biasT = curr.getBias();
        SimpleMatrix dot = prev.mult(weights);
        int rows = dot.getNumRows();
        int cols = dot.getNumCols();
        double[][] bias = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                bias[i][j] = biasT.get(j);
            }
        }

        return dot.plus(new SimpleMatrix(bias));
    }

    public double std(SimpleMatrix v) {
        int numElements = v.getNumElements();
        double mean = (v.elementSum() / numElements);
        double s = 0;

        for (int i = 0; i < numElements; i++) {
            s += (v.get(i) - mean) * (v.get(i) - mean);
        }

        return Math.sqrt(s / numElements);
    }
}
