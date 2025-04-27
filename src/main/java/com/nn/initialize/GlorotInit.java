package com.nn.initialize;

import java.util.Arrays;
import java.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import com.nn.components.*;
import com.nn.layers.Dense;

public class GlorotInit extends InitWeights {
    public INDArray initWeight(Layer prev, Layer curr) {
        return Nd4j.create(setWeights(((Dense)prev).getNumNeurons(), ((Dense)curr).getNumNeurons()));
    }

    public INDArray initWeight(int inputSize, Layer curr) {
        return Nd4j.create(setWeights(inputSize, ((Dense)curr).getNumNeurons()));
    }

    public INDArray initFilters(Layer curr, int[] kernelSize, int numFilters) {
        // slow
        // int preAct = (int)curr.getPreActivation().shape()[2];
        INDArray filters = Nd4j.zeros(numFilters, kernelSize[0], kernelSize[1], curr.getPreActivation().shape()[3]);
        int fan_in = kernelSize[0] * kernelSize[1] * kernelSize[1];
        float varW = (float) (2.0 / (fan_in + numFilters));
        for (int i = 0; i < numFilters; i++) {
            INDArray currFilt = filters.get(NDArrayIndex.interval(i, i + 1));
            for (int j = 0; j < kernelSize[0]; j++) {
                for (int k = 0; k < kernelSize[1]; k++) {
                    for (int m = 0; m < curr.getPreActivation().shape()[3]; m++) {
                        currFilt.getScalar(1, j, k, m).addi((float) (new Random().nextGaussian() * Math.sqrt(varW)));
                    }
                    
                }
                
            }
            
        }
        // int fan_in = inputSize[0] * inputSize[1] * inputSize[2];
        // float varW = (float) (2.0 / (fan_in + numFilters));
        // filters.addi((float) (rand.nextGaussian() * Math.sqrt(varW)));

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
