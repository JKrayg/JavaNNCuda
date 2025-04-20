package com.nn.initialize;

import java.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.nn.components.Layer;
import com.nn.layers.Conv2d;
import com.nn.layers.Dense;

public class HeInit extends InitWeights {
    public INDArray initWeight(Layer prev, Layer curr) {
        if (prev instanceof Dense) {
            return Nd4j.create(setWeights(((Dense)prev).getNumNeurons(), ((Dense)curr).getNumNeurons()));
        } else {
            return Nd4j.create(setWeights(
                (int)(((Conv2d)prev).getActivations().reshape(((Conv2d)prev).getActivations().shape()[0], -1).shape()[1]),
                ((Dense)curr).getNumNeurons()));
        }
        
    }

    public INDArray initWeight(int inputSize, Layer curr) {
        return Nd4j.create(setWeights(inputSize, ((Dense)curr).getNumNeurons()));
    }

    private static float[][] setWeights(int rows, int cols) {
        float std = (float) Math.sqrt(2.0 / rows);
        float[][] weights = new float[rows][cols];
        Random rand = new Random();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                weights[i][j] = (float) (rand.nextGaussian() * std);
            }
        }

        return weights;
    }
}
