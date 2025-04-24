package com.nn.layers;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import com.nn.activation.ActivationFunction;
import com.nn.activation.ReLU;
import com.nn.components.Layer;
import com.nn.initialize.GlorotInit;
import com.nn.initialize.HeInit;
import com.nn.training.normalization.BatchNormalization;

public class Conv2d extends Layer {
    private int[] inputSize;
    private INDArray filters;
    private int numFilters;
    private int[] kernelSize;
    private int stride;
    private int padding;

    public Conv2d(int numFilters, int[] inputSize, int[] kernelSize, int stride,
            int padding, ActivationFunction actFunc) {
        // this.setPreActivations(Nd4j.create(inputSize));
        this.numFilters = numFilters;
        // this.filters = new GlorotInit().initFilters(this, kernelSize, numFilters);
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.inputSize = inputSize;
        this.setActivationFunction(actFunc);
        // this.setBiases(Nd4j.create(this.filters.size(0)));

    }

    public Conv2d(int numFilters, int[] kernelSize, int stride,
            int padding, ActivationFunction actFunc) {
        this.numFilters = numFilters;
        // this.filters = new GlorotInit().initFilters(this, kernelSize, numFilters);
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.setActivationFunction(actFunc);
        // this.setBiases(Nd4j.create(this.filters.size(0)));

    }

    public int[] getInputSize() {
        return ((Conv2d) this).inputSize;
    }

    public int[] getKernelSize() {
        return this.kernelSize;
    }

    public int getStride() {
        return stride;
    }

    public int getPadding() {
        return padding;
    }

    public INDArray getFilters() {
        return filters;
    }

    public void setFilters(INDArray filters) {
        this.filters = filters;
    }

    public INDArray convolve(INDArray data) {
        int kernelWdth = kernelSize[0];
        int actDim = ((((int) this.getPreActivation().shape()[1]) + (2 * padding) - kernelSize[0]) / stride) + 1;
        INDArray acts = Nd4j.create(actDim, actDim, filters.shape()[0]);
        INDArray z;
        for (int k = 1; k < filters.shape()[0]; k++) {
            INDArray fltr = filters.get(NDArrayIndex.interval(k - 1, k));
            for (int j = 1; j < data.shape()[1] - kernelWdth; j += 1) {
                for (int i = 1; i < data.shape()[1] - kernelWdth; i += 1) {
                    // System.out.println("filters shape: " + Arrays.toString(filters.shape()));
                    // System.out.println("fltr shape: " + Arrays.toString(fltr.shape()));
                    // System.out.println("data shape: " + Arrays.toString(data.shape()));
                    INDArray currentDataChunk = data.get(NDArrayIndex.interval(j - 1, j + kernelWdth),
                            NDArrayIndex.interval(i - 1, i + kernelWdth),
                            NDArrayIndex.all());
                    // System.out.println(Arrays.toString(currentDataChunk.shape()));
                }
            }
        }

        return acts;

    }

    public void forwardProp(Layer prev, INDArray data, INDArray labels) {
        INDArray z;
        if (prev == null) {
            z = this.convolve(data);
        } else {
            z = this.convolve(prev.getActivations());
        }

        this.setActivations(z);
    }

    public Layer initLayer(Layer prev) {
        if (prev != null) {
            INDArray prevAct = prev.getActivations();
            this.setPreActivations(prevAct);
        } else {
            this.setPreActivations(Nd4j.create(inputSize));
        }

        // this.filters = new GlorotInit().initFilters(this, kernelSize, numFilters);
        this.setBiases(Nd4j.create(this.filters.size(0)));
            
        return this;

    }

    public void initForAdam() {
    }
}
