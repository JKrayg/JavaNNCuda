package com.nn.initialize;

import java.util.Arrays;
import java.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import com.nn.components.*;
import com.nn.layers.Dense;

public class GlorotInit{
    public INDArray initWeight(Layer prev, Layer curr) {
        return Nd4j.create(setWeights(((Dense)prev).getNumNeurons(), ((Dense)curr).getNumNeurons()));
    }

    public INDArray initWeight(int inputSize, Layer curr) {
        return Nd4j.create(setWeights(inputSize, ((Dense)curr).getNumNeurons()));
    }

    public INDArray initFilters(Layer prev, int[] kernelSize, int numFilters) {
        INDArray filters = Nd4j.zeros(kernelSize);
        int fan_in = kernelSize[0] * kernelSize[1] * kernelSize[2] * kernelSize[3];
        float varW = (float) (2.0 / (fan_in));

        INDArray rands = Nd4j.randn(kernelSize).muli(Math.sqrt(varW));
        filters.addi(rands);

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
