package com.nn.initialize;

import java.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.nn.components.*;

public class GlorotInit extends InitWeights {
    public INDArray initWeight(Layer prev, Layer curr) {
        return Nd4j.create(setWeights(prev.getNumNeurons(), curr.getNumNeurons()));
    }

    public INDArray initWeight(int inputSize, Layer curr) {
        return Nd4j.create(setWeights(inputSize, curr.getNumNeurons()));
    }

    public INDArray initFilters(int[] inputSize, int numFilters) {
        INDArray filters = Nd4j.create(inputSize[0], inputSize[1], inputSize[2], numFilters);
        Random rand = new Random();
        int fan_in = inputSize[0] * inputSize[1] * inputSize[2];
        float varW = (float) (2.0 / (fan_in + numFilters));
        filters.addi((float) (rand.nextGaussian() * Math.sqrt(varW)));

        return filters;
    }

    private static float[][] setWeights(int rows, int cols) {
        float varW = (float) (1.0 / (rows + cols));
        float[][] weights = new float[rows][cols];
        Random rand = new Random();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                weights[i][j] = (float) (rand.nextGaussian() * Math.sqrt(varW));
            }
        }

        return weights;
    }
}
