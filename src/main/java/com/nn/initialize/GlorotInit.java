package com.nn.initialize;

import java.util.Random;
import org.ejml.simple.SimpleMatrix;
import com.nn.components.*;

public class GlorotInit extends InitWeights {
    public SimpleMatrix initWeight(Layer prev, Layer curr) {
        return new SimpleMatrix(setWeights(prev.getNumNeurons(), curr.getNumNeurons()));
    }

    public SimpleMatrix initWeight(int inputSize, Layer curr) {
        return new SimpleMatrix(setWeights(inputSize, curr.getNumNeurons()));
    }

    private static double[][] setWeights(int rows, int cols) {
        double varW = 1.0 / (rows + cols);
        double[][] weights = new double[rows][cols];
        Random rand = new Random();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                weights[i][j] = rand.nextGaussian() * Math.sqrt(varW);
            }
        }

        return weights;
    }
}
