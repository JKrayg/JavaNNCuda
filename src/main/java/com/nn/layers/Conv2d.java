package com.nn.layers;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu.pad;
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

    public INDArray padImages(INDArray images) {
        INDArray padded = Nd4j.zeros(32,
                                    images.size(1) + padding * 2,
                                    images.size(2) + padding * 2,
                                    images.size(3));
        padded.get(
            NDArrayIndex.all(),
            NDArrayIndex.interval(1, 29),
            NDArrayIndex.interval(1, 29),
            NDArrayIndex.all()
        ).assign(images);

        return padded;
    
    }

    public INDArray convolve(INDArray data) {
        System.out.println(Arrays.toString(data.shape()));
        INDArray images = data;
        if (padding != 0) {
            images = padImages(data);
        }

        
        
        long[] imgShape = images.shape();
        long[] filtersShape = filters.shape();
        int outShapeH = (int) Math.floor(((imgShape[1] + (2 * padding) - filtersShape[1]) / stride) + 1);
        int outShapeW = (int) Math.floor(((imgShape[2] + (2 * padding) - filtersShape[2]) / stride) + 1);
        int patchSize = (int) (filtersShape[0] * filtersShape[1] * filtersShape[2]);
        INDArray patches = Nd4j.create(patchSize, outShapeH * outShapeW);
        // System.out.println(outShape);
        // INDArray activation = Nd4j.create(imgShape[0], outShape, outShape, filtersShape[0]);
        // INDArray patches = Nd4j.create(imgShape[0], outShape, outShape, filtersShape[1], filtersShape[2], filtersShape[3]);
        // // int iter = Math.floor()

        for (int i = 0; i < outShapeH - filtersShape[1]; i+= stride) {
            // INDArray currImg = images.get(NDArrayIndex.interval(i, i + 1));
        //     long[] currImgShape = currImg.shape();
        //     currImg = currImg.reshape(currImgShape[1], currImgShape[2], currImgShape[3]);
                for (int k = 0; k < outShapeW - filtersShape[2]; k += stride) {
                    INDArray currPatch = images.get(
                        NDArrayIndex.all(),
                        NDArrayIndex.interval(i, i + filtersShape[1]),
                        NDArrayIndex.interval(k, k + filtersShape[2]));
                        
                    System.out.println(Arrays.toString(currPatch.shape()));
                }
        }

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
