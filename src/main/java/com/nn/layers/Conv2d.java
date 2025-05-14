package com.nn.layers;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu.pad;
import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu.pointwise_conv2d;
import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu.shape_of;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
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
        
        this.numFilters = numFilters;

        int[] k = new int[]{inputSize[0], kernelSize[0], kernelSize[1]};
        this.kernelSize = k;
        this.stride = stride;

        if (padding.equals("valid")) {
            this.padding = 0;
        } else {
            int in = inputSize[2];
            this.padding = (in * stride - in + kernelSize[1] - stride) / 2;
        }

        this.inputSize = inputSize;
        this.setActivationFunction(actFunc);
        this.setBiases(Nd4j.create(numFilters));

    }

    public Conv2d(int numFilters, int[] kernelSize, int stride,
            String padding, ActivationFunction actFunc) {
        
        this.numFilters = numFilters;
        this.kernelSize = kernelSize;
        this.stride = stride;
        
        if (padding.equals("valid")) {
            this.padding = 0;
        } else {
            this.padding = -1;
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

    public INDArray padData(INDArray data) {
        INDArray padded = Nd4j.zeros(data.size(0),
            data.size(1),
            data.size(2) + padding * 2,
            data.size(3) + padding * 2);
        padded.get(
            NDArrayIndex.all(),
            NDArrayIndex.all(),
            NDArrayIndex.interval(padding, (data.size(2) + padding)),
            NDArrayIndex.interval(padding, (data.size(3) + padding))
        ).assign(data);

        return padded;
    
    }



    public INDArray convolve(INDArray data) {
        long[] dataShape = data.shape();
        INDArray paddedData = data.dup();

        // add padding
        if (padding != 0) {
            paddedData = padData(data);
        }

        long[] paddedShape = paddedData.shape();
        int batchSize = (int) paddedShape[0];
        int outShapeH = (int) Math.floor(((dataShape[2] + (2 * padding) - kernelSize[1]) / stride) + 1);
        int outShapeW = (int) Math.floor(((dataShape[3] + (2 * padding) - kernelSize[2]) / stride) + 1);
        int patchLen = (int) (kernelSize[0] * kernelSize[1] * kernelSize[2]);
        int imgLen = (int) (dataShape[2] * dataShape[3]);
        INDArray patches = Nd4j.create(batchSize, imgLen, patchLen);


        int count = 0;

        for (int i = 0; i < outShapeH; i += stride) {
            for (int k = 0; k < outShapeW; k += stride) {

                INDArray currPatch = paddedData.get(
                        NDArrayIndex.all(),
                        NDArrayIndex.all(),
                        NDArrayIndex.interval(i, i + kernelSize[1]),
                        NDArrayIndex.interval(k, k + kernelSize[2])).reshape(paddedShape[0], -1);

                
                patches.put(
                        new INDArrayIndex[] {
                                NDArrayIndex.all(),
                                NDArrayIndex.point(count),
                                NDArrayIndex.all()
                        }, currPatch);

                count += 1;
            }
        }


        long[] fShape = filters.shape();
        INDArray reShapefilters = filters.reshape(fShape[0], -1).transpose();
        INDArray out = Nd4j.create(patches.shape()[0], numFilters, outShapeH, outShapeW);

        for (int i = 0; i < patches.shape()[0]; i++) {
            INDArray currImgPatches = patches.get(NDArrayIndex.point(i));
            
            INDArrayIndex[] slice = new INDArrayIndex[] {
                NDArrayIndex.point(i),
                NDArrayIndex.all(),
                NDArrayIndex.all()};

            
            INDArray col = currImgPatches.mmul(reShapefilters);
            out.put(slice, col.reshape(numFilters, outShapeH, outShapeW));
        }

        System.out.println(Arrays.toString(out.shape()));

        return out;

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

    public Layer initLayer(Layer prev, int batchSize) {

        int actDim;
        if (prev != null) {

            long[] inShape = prev.getActivations().shape();
            if (this.padding == -1) {
                this.padding = (int)(inShape[2] * stride - inShape[3] + kernelSize[1] - stride) / 2;
            }

            this.kernelSize = new int[]{(int)inShape[1], kernelSize[0], kernelSize[1]};
            actDim = (int)((inShape[2] + (2 * this.padding) - kernelSize[1]) / stride) + 1;
            this.filters = new GlorotInit().initFilters(prev, kernelSize, numFilters);

        } else {

            actDim = ((inputSize[1] + (2 * this.padding) - kernelSize[1]) / stride) + 1;
            prev = new Layer();
            prev.setActivations(Nd4j.create(batchSize, inputSize[0], inputSize[1], inputSize[2]));
            this.filters = new GlorotInit().initFilters(prev, kernelSize, numFilters);

        }

        int[] actShape = new int[]{batchSize, numFilters, actDim, actDim};
        this.setPreActivations(Nd4j.create(actShape));
        this.setActivations(Nd4j.create(actShape));
            
        return this;

    }

    public void initForAdam() {
    }
}
