package com.nn.layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.nn.activation.ActivationFunction;
import com.nn.components.Layer;
import com.nn.initialize.GlorotInit;

public class Conv2d extends Layer {
    private int[] inputSize;
    private INDArray filters;
    private int[] kernelSize;
    private int stride;
    private int padding;
    

    public Conv2d(int[] inputSize, int filters, int[] kernelSize, int stride,
                  int padding, ActivationFunction actFunc) {
        this.filters = new GlorotInit().initFilters(inputSize, filters);
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.setActivationFunction(actFunc);
        this.inputSize = inputSize;

        int actDim = ((inputSize[0] + (2*padding) - kernelSize[0]) / stride) + 1;
        this.setActivations(Nd4j.create(actDim, actDim));
        this.setBiases(Nd4j.create(this.filters.length()));

    }
}
