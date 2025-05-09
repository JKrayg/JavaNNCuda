package com.nn.layers;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu.pad;
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
        // this.setPreActivations(Nd4j.create(inputSize));
        this.numFilters = numFilters;
        // this.filters = new GlorotInit().initFilters(this, kernelSize, numFilters);
        int[] k = new int[]{inputSize[0], kernelSize[0], kernelSize[1]};
        this.kernelSize = k;
        this.stride = stride;
        if (padding.equals("valid")) {
            this.padding = 0;
        } else {
            int in = inputSize[2];
            this.padding = (in * stride - in + kernelSize[1] - stride) / 2;
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
        // int[] k = new int[]{numFilters, kernelSize[0], kernelSize[1]};
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
        // System.out.println(Arrays.toString(paddedData.shape()));

        // add padding
        if (padding != 0) {
            paddedData = padData(data);
        }

        long[] paddedShape = paddedData.shape();
        int batchSize = (int) paddedShape[0];
        // long[] kernelShape = filters.shape();
        // System.out.println("+++++++++++++: " + Arrays.toString(paddedShape));
        int outShapeH = (int) Math.floor(((paddedShape[2] + (2 * padding) - kernelSize[1]) / stride) + 1);
        int outShapeW = (int) Math.floor(((paddedShape[3] + (2 * padding) - kernelSize[2]) / stride) + 1);
        int patchShape = (int) (kernelSize[0] * kernelSize[1] * kernelSize[2]);
        INDArray patches = Nd4j.create(batchSize, dataShape[2] * dataShape[3], kernelSize[0], kernelSize[1], kernelSize[2]);
        // System.out.println(padding);
        // System.out.println("---: " + Arrays.toString(paddedData.shape()));
        // INDArray currPatch = paddedData.get(
        //                 NDArrayIndex.all(),
        //                 NDArrayIndex.interval(10, 13),
        //                 NDArrayIndex.interval(10, 13),
        //                 NDArrayIndex.point(3)
        //             );
        
        // System.out.println("-------------: " + currPatch);

        int count = 0;

        for (int i = 0; i < /*dataShape[1]*/ 1; i += stride) {
                for (int k = 0; k < /*dataShape[2]*/ 1; k += stride) {

                    INDArray currPatch = paddedData.get(
                        NDArrayIndex.all(),
                        NDArrayIndex.all(),
                        NDArrayIndex.interval(i, i + kernelSize[1]),
                        NDArrayIndex.interval(k, k + kernelSize[2]));

                    patches.put(
                        new INDArrayIndex[] {
                            NDArrayIndex.all(),
                            NDArrayIndex.point(count),
                            NDArrayIndex.all(),
                            NDArrayIndex.all(),
                            NDArrayIndex.all()
                        }, currPatch);

                    count += 1;
                }
        }

        long[] pShape = patches.shape();
        long[] fShape = filters.shape();
        patches = patches.reshape(pShape[0], pShape[1], -1);
        INDArray reShapefilters = filters.reshape(fShape[0], -1).transpose();


        // INDArray output = Nd4j.batchedGEMM(patches, filters);


        System.out.println("---: " + Arrays.toString(patches.shape()));
        System.out.println("+++: " + Arrays.toString(reShapefilters.shape()));

        int a;

        if (data.shape()[1] == 1) {
            a = (int) 10;
            return data.repeat(1, a);
        } else if (data.shape()[1] == 10){
            a = (int) 2;
            return data.repeat(1, a);
        } else {
            return data;
        }

    }

    public void forwardProp(Layer prev, INDArray data, INDArray labels) {
        // System.out.println("Data: " + Arrays.toString(data.shape()));
        INDArray z;
        if (prev == null) {
            z = this.convolve(data);
            // System.out.println("z1: " + Arrays.toString(z.shape()));
        } else {
            z = this.convolve(prev.getActivations());
            // System.out.println("z2: " + Arrays.toString(z.shape()));
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
                this.padding = (int)(inShape[2] * stride - inShape[3] + kernelSize[1] - stride) / 2;
            }
            this.kernelSize = new int[]{(int)inShape[1], kernelSize[0], kernelSize[1]};
            actDim = (int)((inShape[2] + (2 * this.padding) - kernelSize[1]) / stride) + 1;
            this.filters = new GlorotInit().initFilters(prev, kernelSize, numFilters);
            // int[] actShape = new int[]{batchSize, actDim, actDim, numFilters};
            // this.setActivations(Nd4j.create(actShape));
        } else {
            // this.setPreActivations(Nd4j.create(batchSize, inputSize[0], inputSize[1], inputSize[2]));
            // this.filters = new GlorotInit().initFilters(this, kernelSize, numFilters);
            actDim = ((inputSize[1] + (2 * this.padding) - kernelSize[1]) / stride) + 1;
            prev = new Layer();
            prev.setActivations(Nd4j.create(batchSize, inputSize[0], inputSize[1], inputSize[2]));
            this.filters = new GlorotInit().initFilters(prev, kernelSize, numFilters);
            // int[] actShape = new int[]{batchSize, actDim, actDim, numFilters};
            // this.setActivations(Nd4j.create(actShape));
        }

        int[] actShape = new int[]{batchSize, numFilters, actDim, actDim};
        this.setPreActivations(Nd4j.create(actShape));
        // this.filters = new GlorotInit().initFilters(prev, kernelSize, numFilters);
        this.setActivations(Nd4j.create(actShape));
            
        return this;

    }

    public void initForAdam() {
    }
}
