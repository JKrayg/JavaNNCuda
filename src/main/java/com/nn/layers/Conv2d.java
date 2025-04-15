package com.nn.layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import com.nn.activation.ActivationFunction;
import com.nn.components.Layer;
import com.nn.initialize.GlorotInit;

public class Conv2d extends Layer {
    private INDArray filters;
    private int[] kernelSize;
    private int[] stride;
    private String padding;
    private int[] inputSize;

    public Conv2d(int filters, int[] kernelSize, int[] stride,
                  String padding, ActivationFunction actFunc, int[] inputSize) {
        this.filters = new GlorotInit().initFilters(inputSize, filters);
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.setActivationFunction(actFunc);
        this.inputSize = inputSize;

    }
}
