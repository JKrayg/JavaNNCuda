package com.nn.layers;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Col2Im;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu.pad;
import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu.pointwise_conv2d;
import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu.shape_of;
import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu.softplus;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDCNN;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import com.nn.activation.ActivationFunction;
import com.nn.activation.ReLU;
import com.nn.components.Layer;
import com.nn.initialize.GlorotInit;
import com.nn.initialize.HeInit;
import com.nn.training.normalization.BatchNormalization;
import com.nn.training.normalization.Normalization;
import com.nn.training.optimizers.Optimizer;
import com.nn.training.regularizers.L1;
import com.nn.training.regularizers.L2;
import com.nn.training.regularizers.Regularizer;

public class Conv2d extends Layer {
    private int[] inputSize;
    private INDArray filters;
    private INDArray filtersGradient;
    private int numFilters;
    private int[] kernelSize;
    private int stride;
    private int padding;
    private String typePadding;
    // private int[] actShape;

    public Conv2d(int numFilters, int[] inputSize, int[] kernelSize, int stride,
            String padding, ActivationFunction actFunc) {
        
        this.numFilters = numFilters;

        int[] k = new int[]{numFilters, inputSize[0], kernelSize[0], kernelSize[1]};
        this.kernelSize = k;
        this.stride = stride;
        this.typePadding = padding;

        if (padding.equals("valid")) {
            this.padding = 0;
        } else {
            int in = inputSize[2];
            this.padding = (in * stride - in + kernelSize[1] - stride) / 2;
        }

        this.inputSize = inputSize;
        this.setActivationFunction(actFunc);
        this.setBiases(Nd4j.create(numFilters, 1));

    }

    public Conv2d(int numFilters, int[] kernelSize, int stride,
            String padding, ActivationFunction actFunc) {
        
        this.numFilters = numFilters;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.typePadding = padding;
        
        if (padding.equals("valid")) {
            this.padding = 0;
        } else {
            this.padding = -1;
        }

        this.setActivationFunction(actFunc);
        this.setBiases(Nd4j.create(numFilters, 1));

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

    public INDArray getWeights() {
        return this.filters;
    }

    public void initForAdam() {
        INDArray weightsO = Nd4j.create(filters.reshape(filters.shape()[0], -1).shape());
        // System.out.println("Weights O: " + Arrays.toString(weightsO.shape()));
        INDArray biasO = Nd4j.create(this.getBias().shape());

        this.setWeightsMomentum(weightsO);
        this.setWeightsVariance(weightsO);
        this.setBiasesMomentum(biasO);
        this.setBiasesVariance(biasO);

        Normalization norm = this.getNormalization();
        if (norm != null) {
            INDArray shiftO = Nd4j.create(norm.getShift().shape());
            INDArray scaleO = Nd4j.create(norm.getScale().shape());

            norm.setShiftMomentum(shiftO);
            norm.setShiftVariance(shiftO);
            norm.setScaleMomentum(scaleO);
            norm.setScaleVariance(scaleO);
        }
    }

    public void updateWeights(Optimizer o) {
        this.filters = o.executeWeightsUpdate(this);
    }

    public Layer initLayer(Layer prev, int batchSize) {

        int actDim;
        if (prev != null) {

            long[] inShape = prev.getActivations().shape();
            if (this.padding == -1) {
                this.padding = (int)(inShape[2] * stride - inShape[3] + kernelSize[1] - stride) / 2;
            }

            this.kernelSize = new int[]{numFilters, (int)inShape[1], kernelSize[0], kernelSize[1]};
            actDim = (int)((inShape[2] + (2 * this.padding) - kernelSize[2]) / stride) + 1;
            this.filters = new GlorotInit().initFilters(prev, kernelSize, numFilters);

        } else {

            actDim = ((inputSize[1] + (2 * this.padding) - kernelSize[2]) / stride) + 1;
            prev = new Layer();
            prev.setActivations(Nd4j.create(batchSize, inputSize[0], inputSize[1], inputSize[2]));
            System.out.println("-------: " + Arrays.toString(prev.getActivations().shape()));
            this.filters = new GlorotInit().initFilters(prev, kernelSize, numFilters);

        }

        int[] actShape = new int[]{batchSize, numFilters, actDim, actDim};
        this.setPreActivations(Nd4j.create(actShape));
        this.setActivations(Nd4j.create(actShape));
            
        return this;

    }


    public INDArray padData(INDArray data, int padding) {
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

    public INDArray im2col(INDArray data) {
        long[] dataShape = data.shape();
        int batchSize = (int) dataShape[0];
        int inChannels = (int) dataShape[1];

        if (padding != 0) {
            data = padData(data, this.padding);
        }

        int outShapeH = (int) Math.floor(((dataShape[2] + (2 * padding) - kernelSize[2]) / stride) + 1);
        int outShapeW = (int) Math.floor(((dataShape[3] + (2 * padding) - kernelSize[3]) / stride) + 1);
        int patchLen = (int) (inChannels * kernelSize[2] * kernelSize[3]);
        int imgLen = (int) (outShapeH * outShapeW);
        INDArray patches = Nd4j.create(batchSize, inChannels, kernelSize[2], kernelSize[3], imgLen);
        


        int patchIndex = 0;

        for (int i = 0; i < outShapeH; i += stride) {
            for (int k = 0; k < outShapeW; k += stride) {
                INDArray currPatch = data.get(
                        NDArrayIndex.all(),
                        NDArrayIndex.all(),
                        NDArrayIndex.interval(i, i + kernelSize[2]),
                        NDArrayIndex.interval(k, k + kernelSize[3]));

                patches.put(
                        new INDArrayIndex[] {
                                NDArrayIndex.all(),
                                NDArrayIndex.all(),
                                NDArrayIndex.all(),
                                NDArrayIndex.all(),
                                NDArrayIndex.point(patchIndex)
                        }, currPatch);

                patchIndex += 1;
            }
        }

        INDArray reshPatches = patches.reshape(-1, patchLen);

        return reshPatches;
    }


    public INDArray convolve(INDArray data) {
        long[] dataShape = data.shape();
        int outShapeH = (int) Math.floor(((dataShape[2] + (2 * padding) - kernelSize[2]) / stride) + 1);
        int outShapeW = (int) Math.floor(((dataShape[3] + (2 * padding) - kernelSize[3]) / stride) + 1);
        INDArray patches = im2col(data);
        long[] fShape = filters.shape();
        INDArray out = patches.mmul(filters.reshape(-1, fShape[0])).reshape(data.shape()[0], numFilters, outShapeH, outShapeW);

        return out;

    }

    public void forwardProp(Layer prev) {
        INDArray z = this.convolve(prev.getActivations());
        // System.out.println("z: " + Arrays.toString(z.shape()));
        this.setActivations(z);
    }

    public INDArray gradientFilters(INDArray activations, INDArray gradient) {
        INDArray gWrtW = activations.mmul(gradient).div(activations.rows());
        return gWrtW;
    }

    public INDArray col2im(INDArray gradient) {
        System.out.println("incoming gradient: " + Arrays.toString(gradient.shape()));
        INDArray g = Nd4j.create(this.getPrev().getActivations().shape());
        // System.out.println("prev act shape: " + Arrays.toString(g.shape()));
        // System.out.println("patch size: " + Arrays.toString(gradient.get(NDArrayIndex.point(0)).shape()));
        // System.out.println("kernel size: " + Arrays.toString(kernelSize));
        for (int i = 0; i < gradient.columns(); i++) {
            INDArray currPatch = gradient.get(NDArrayIndex.all(), NDArrayIndex.point(i)).reshape(kernelSize);
        }
        return gradient;
    }

    public void getGradients(Layer prev, INDArray gradient, INDArray data) {
        // im2col prev activations
        // reshape gradient
        // get gradient weights
        // get gradient bias
        // System.out.println("gradient: " + Arrays.toString(gradient.shape()));
        // System.out.println("kernel: " + Arrays.toString(kernelSize));

        
        if (prev != null) {
            // System.out.println("input shape: " + Arrays.toString(prev.getActivations().shape()));
            INDArray act = im2col(prev.getActivations());
            INDArray grad = gradient.reshape(-1, gradient.shape()[1]);
            // System.out.println("gradient wrt weights: " + Arrays.toString(this.gradientWeights(act, grad).shape()));
            // System.out.println("gradient wrt bias: " + Arrays.toString(this.gradientBias(grad).shape()));
            // System.out.println("asd: " + Arrays.toString(this.gradientWeights(act, grad).reshape(this.getWeights().shape()).shape()));
            this.setGradientWeights(this.gradientWeights(act, grad));
            this.setGradientBiases(this.gradientBias(grad));

            INDArray weights = this.getWeights();
            weights = weights.reshape(weights.shape()[0], -1);
            // System.out.println("we: " + Arrays.toString(weights.shape()));

            this.padding = 1;
            if (typePadding == "valid") {
                gradient = padData(gradient, 1);
            }

            // System.out.println("p: " + Arrays.toString(prev.getPreActivation().shape()));
            // System.out.println("w: " + Arrays.toString(gradient.reshape(-1, gradient.shape()[1]).shape()));
            // System.out.println("wr: " + Arrays.toString(gradient.reshape(-1, gradient.shape()[1]).mmul(weights).shape()));
            // INDArray gWrtPreAct = gradient.reshape(-1, gradient.shape()[1]).mmul(weights);
            // INDArray next = prev.getActFunc().gradient(
            //     im2col(prev.getPreActivation()),
            //     gradient.reshape(-1, gradient.shape()[1]).mmul(weights));


            INDArray gWrtPreAct = col2im(gradient.reshape(-1, gradient.shape()[1]).mmul(weights));

            // System.out.println("col2im: " + Arrays.toString(gWrtPreAct.shape()));
            
            
            
            // System.out.println("gradient passed to next layer c: " + Arrays.toString(next.shape()));

            // prev.getGradients(prev.getPrev(), next, data);

        } else {
            // System.out.println("input shape: " + Arrays.toString(data.shape()));
            // this.padding = 0;
            INDArray i2cAct = im2col(data);
            INDArray i2cGrad = gradient.reshape(-1, gradient.shape()[1]);
            this.setGradientWeights(this.gradientWeights(i2cAct, i2cGrad));
            this.setGradientBiases(this.gradientBias(i2cGrad));
        }
        

    }
    
}
