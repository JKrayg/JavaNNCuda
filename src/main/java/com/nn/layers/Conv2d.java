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
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.setActivationFunction(actFunc);
        this.inputSize = inputSize;

        // int actDim = ((inputSize[0] + (2*padding) - kernelSize[0]) / stride) + 1;
        // this.setActivations(Nd4j.create(filters, actDim, actDim));
        this.setBiases(Nd4j.create(this.filters.size(0)));

    }

    public Conv2d(int filters, int[] kernelSize, int stride,
                  int padding, ActivationFunction actFunc) {
        this.filters = new GlorotInit().initFilters(kernelSize, filters);
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.setActivationFunction(actFunc);

        // int actDim = ((25 + (2*padding) - kernelSize[0]) / stride) + 1;
        // this.setActivations(Nd4j.create(actDim, actDim));
        this.setBiases(Nd4j.create(this.filters.size(0)));

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
        int actDim = ((inputSize[0] + (2*padding) - kernelSize[0]) / stride) + 1;
        INDArray acts = Nd4j.create(filters.shape()[0], actDim, actDim);
        INDArray z;
        for (int k = 1; k < filters.shape()[0]; k++) {
            INDArray fltr = filters.get(NDArrayIndex.interval(k - 1, k));
            for (int j = 1 ; j < data.shape()[1] - kernelWdth; j += 1) {
                    for (int i = 1; i < data.shape()[1] - kernelWdth; i += 1) {
                        // System.out.println("filters shape: " + Arrays.toString(filters.shape()));
                        // System.out.println("fltr shape: " + Arrays.toString(fltr.shape()));
                        // System.out.println("data shape: " + Arrays.toString(data.shape()));
                        INDArray currentDataChunk = data.get(NDArrayIndex.all(),
                                                            NDArrayIndex.interval(j - 1, j + kernelWdth),
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

    public void initForAdam() {
        
    }
}
