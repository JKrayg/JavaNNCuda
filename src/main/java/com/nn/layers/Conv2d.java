package com.nn.layers;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import com.nn.activation.ActivationFunction;
import com.nn.components.Layer;
import com.nn.initialize.GlorotInit;

public class Conv2d extends Layer {
    private int[] inputSize;
    private INDArray filters;
    private int[] kernelSize;
    private int stride;
    private int padding;
    

    public Conv2d(int filters, int[] inputSize, int[] kernelSize, int stride,
                  int padding, ActivationFunction actFunc) {
        this.filters = new GlorotInit().initFilters(kernelSize, filters);
        System.out.println(Arrays.toString(this.filters.shape()));
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.setActivationFunction(actFunc);
        this.inputSize = inputSize;

        int actDim = ((inputSize[0] + (2*padding) - kernelSize[0]) / stride) + 1;
        this.setActivations(Nd4j.create(actDim, actDim));
        this.setBiases(Nd4j.create(this.filters.length()));

    }

    public Conv2d(int filters, int[] kernelSize, int stride,
                  int padding, ActivationFunction actFunc) {
        this.filters = new GlorotInit().initFilters(kernelSize, filters);
        System.out.println(Arrays.toString(this.filters.shape()));
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.setActivationFunction(actFunc);

        int actDim = ((25 + (2*padding) - kernelSize[0]) / stride) + 1;
        this.setActivations(Nd4j.create(actDim, actDim));
        this.setBiases(Nd4j.create(this.filters.length()));

    }

    public int[] getInputSize() {
        return ((Conv2d)this).inputSize;
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
        INDArray z;
        for (int j = 0 ; j < data.shape()[1] - kernelWdth; j += stride) {
                for (int i = 0; i < data.shape()[1] - kernelWdth; i += stride) {
                    System.out.println("filters shape: " + Arrays.toString(filters.shape()));
                    INDArray fltr = filters.get(NDArrayIndex.interval(i, i + 1));
                    System.out.println("fltr shape: " + Arrays.toString(fltr.shape()));
                    System.out.println("data shape: " + Arrays.toString(data.shape()));
                    INDArray currentDataChunk = data.get(NDArrayIndex.all(),
                                                        NDArrayIndex.interval(j, j + kernelWdth),
                                                        NDArrayIndex.interval(i, i + kernelWdth),
                                                        NDArrayIndex.all());
                    System.out.println(Arrays.toString(currentDataChunk.shape()));
            }
        }
        

        return this.getActivations();
        
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

    public void initForAdam() {
        
    }
}
