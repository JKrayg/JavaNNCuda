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
    // private int[] actShape;

    public Conv2d(int numFilters, int[] inputSize, int[] kernelSize, int stride,
            String padding, ActivationFunction actFunc) {
        // this.setPreActivations(Nd4j.create(inputSize));
        this.numFilters = numFilters;
        // this.filters = new GlorotInit().initFilters(this, kernelSize, numFilters);
        this.kernelSize = kernelSize;
        this.stride = stride;
        if (padding.equals("valid")) {
            this.padding = 0;
        } else {
            int in = inputSize[1];
            this.padding = (in * stride - in + kernelSize[0] - stride) / 2;
            // this.padding = Math.max(((in / stride) - 1) * stride + (kernelSize[0]) - in, 0);
        }
        // int actDim = ((inputSize[1] + (2 * this.padding) - kernelSize[0]) / stride) + 1;
        // int[] actShape = new int[]{inputSize[0], actDim, actDim, numFilters};
        // this.setActivations(Nd4j.create(actShape));
        this.inputSize = inputSize;
        this.setActivationFunction(actFunc);
        this.setBiases(Nd4j.create(numFilters));

    }

    public Conv2d(int numFilters, int[] kernelSize, int stride,
            String padding, ActivationFunction actFunc) {
        this.numFilters = numFilters;
        // this.filters = new GlorotInit().initFilters(this, kernelSize, numFilters);
        this.kernelSize = kernelSize;
        this.stride = stride;
        if (padding.equals("valid")) {
            this.padding = 0;
        } else {
            this.padding = -1;
            // int in = inputSize[1] * inputSize[2];
            // this.padding = Math.max(((in / stride) - 1) * stride + (kernelSize[0]*kernelSize[1]) - in, 0);
            // System.out.println("padding: " + this.padding);
        }
        this.setActivationFunction(actFunc);
        this.setBiases(Nd4j.create(numFilters));

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

    // public int[] getActShape() {
    //     return actShape;
    // }

    public void setFilters(INDArray filters) {
        this.filters = filters;
    }

    public INDArray convolve(INDArray data) {
        // int kernelWdth = kernelSize[0];
        // int actDim = ((((int) this.getPreActivation().shape()[1]) + (2 * padding) - kernelSize[0]) / stride) + 1;
        // INDArray acts = Nd4j.create((int) this.getPreActivation().shape()[0], actDim, actDim, filters.shape()[0]);
        // INDArray z;
        // for (int k = 1; k < filters.shape()[0]; k++) {
        //     INDArray fltr = filters.get(NDArrayIndex.interval(k - 1, k));
        //     for (int j = 1; j < data.shape()[1] - kernelWdth; j += 1) {
        //         for (int i = 1; i < data.shape()[1] - kernelWdth; i += 1) {
        //             // System.out.println("filters shape: " + Arrays.toString(filters.shape()));
        //             // System.out.println("fltr shape: " + Arrays.toString(fltr.shape()));
        //             // System.out.println("data shape: " + Arrays.toString(data.shape()));
        //             INDArray currentDataChunk = data.get(NDArrayIndex.interval(j - 1, j + kernelWdth),
        //                     NDArrayIndex.interval(i - 1, i + kernelWdth),
        //                     NDArrayIndex.all());
        //             // System.out.println(Arrays.toString(currentDataChunk.shape()));
        //         }
        //     }
        // }

        return this.getActivations();

    }

    public void forwardProp(Layer prev, INDArray data, INDArray labels) {
        INDArray z;
        if (prev == null) {
            // this.setPreActivations(data);
            z = this.convolve(data);
        } else {
            // this.setPreActivations(prev.getActivations());
            z = this.convolve(prev.getActivations());
        }

        this.setActivations(z);
    }

    public Layer initLayer(Layer prev, int batchSize) {
        
        // if (prev != null) {
        //     System.out.println("not null");
        //     INDArray prevAct = prev.getActivations();
        //     System.out.println(Arrays.toString(prevAct.shape()));
        //     this.setPreActivations(prevAct);
        //     int actDim = ((((int) prevAct.shape()[1]) + (2 * padding) - kernelSize[0]) / stride) + 1;
        //     INDArray acts = Nd4j.create((int) prevAct.shape()[0], actDim, actDim, filters.shape()[0]);
        //     this.setActivations(acts);
        // } else {
        //     System.out.println("null");
        //     this.setPreActivations(Nd4j.create(inputSize));
        //     int actDim = ((((int) inputSize[1]) + (2 * padding) - kernelSize[0]) / stride) + 1;
        //     INDArray acts = Nd4j.create((int) inputSize[0], actDim, actDim, filters.shape()[0]);
        //     this.setActivations(acts);
        //     this.filters = new GlorotInit().initFilters(this, kernelSize, numFilters);
        //     this.setBiases(Nd4j.create(this.filters.size(0)));
        // }

        int actDim;
        if (prev != null) {
            // this.setPreActivations(prev.getActivations());
            // this.filters = new GlorotInit().initFilters(this, kernelSize, numFilters);
            long[] inShape = prev.getActivations().shape();
            if (this.padding == -1) {
                this.padding = (int)(inShape[1] * stride - inShape[1] + kernelSize[0] - stride) / 2;
            }
            actDim = (int)((inShape[1] + (2 * this.padding) - kernelSize[0]) / stride) + 1;
            this.filters = new GlorotInit().initFilters(prev, kernelSize, numFilters);
            // int[] actShape = new int[]{batchSize, actDim, actDim, numFilters};
            // this.setActivations(Nd4j.create(actShape));
        } else {
            // this.setPreActivations(Nd4j.create(batchSize, inputSize[0], inputSize[1], inputSize[2]));
            // this.filters = new GlorotInit().initFilters(this, kernelSize, numFilters);
            actDim = ((inputSize[1] + (2 * this.padding) - kernelSize[0]) / stride) + 1;
            prev = new Layer();
            prev.setActivations(Nd4j.create(batchSize, inputSize[0], inputSize[1], inputSize[2]));
            this.filters = new GlorotInit().initFilters(prev, kernelSize, numFilters);
            // int[] actShape = new int[]{batchSize, actDim, actDim, numFilters};
            // this.setActivations(Nd4j.create(actShape));
        }

        int[] actShape = new int[]{batchSize, actDim, actDim, numFilters};
        this.setPreActivations(Nd4j.create(actShape));
        // this.filters = new GlorotInit().initFilters(prev, kernelSize, numFilters);
        this.setActivations(Nd4j.create(actShape));
            
        return this;

    }

    public void initForAdam() {
    }
}
