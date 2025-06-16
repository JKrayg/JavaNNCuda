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
        // slow
        // int preAct = (int)curr.getPreActivation().shape()[2];
        INDArray filters = Nd4j.zeros(numFilters, kernelSize[0], kernelSize[1], kernelSize[2]);
        // System.out.println("filters: " + Arrays.toString(filters.shape()));
        int fan_in = kernelSize[0] * kernelSize[1] * kernelSize[2];
        float varW = (float) (2.0 / (fan_in + numFilters));
        for (int i = 0; i < numFilters; i++) {
            INDArray currFilt = filters.get(NDArrayIndex.interval(i, i + 1));
            // System.out.println("currFilt: " + Arrays.toString(currFilt.shape()));
            for (int j = 0; j < kernelSize[1]; j++) {
                for (int k = 0; k < kernelSize[2]; k++) {
                    // System.out.println("prevAct: " + Arrays.toString(prev.getActivations().shape()));
                    for (int m = 0; m < prev.getActivations().shape()[1]; m++) {
                        currFilt.getScalar(1, m, j, k).addi((float) (new Random().nextGaussian() * Math.sqrt(varW)));
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
