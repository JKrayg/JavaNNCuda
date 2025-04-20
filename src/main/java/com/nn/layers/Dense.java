package com.nn.layers;

import java.util.ArrayList;
import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.nn.activation.ActivationFunction;
import com.nn.components.*;
import com.nn.training.normalization.BatchNormalization;
import com.nn.training.normalization.Normalization;
import com.nn.training.optimizers.Optimizer;
import com.nn.training.regularizers.Dropout;
import com.nn.training.regularizers.Regularizer;
import com.nn.utils.MathUtils;

// OutputLayer and Dense are pretty much the same class
public class Dense extends Layer {
    private int numNeurons;
    private INDArray weights;
    private INDArray weightsMomentum;
    private INDArray weightsVariance;
    private INDArray gradientWrtWeights;
    private int numFeatures;
    private int inputSize;

    public Dense(int numNeurons, ActivationFunction actFunc) {
        super(actFunc);
        this.numNeurons = numNeurons;
    }

    public Dense(int numNeurons, ActivationFunction actFunc, int inputSize) {
        super(actFunc, inputSize);
        this.numNeurons = numNeurons;
        this.numFeatures = inputSize;
    }

    public int getNumFeatures() {
        return numFeatures;
    }

    public INDArray getWeightsMomentum() {
        return weightsMomentum;
    }

    public INDArray getWeightsVariance() {
        return weightsVariance;
    }

    public INDArray getWeights() {
        return weights;
    }

    public int getNumNeurons() {
        return numNeurons;
    }

    public void setGradientWeights(INDArray gWrtW) {
        this.gradientWrtWeights = gWrtW;
    }

    public void setWeightsVariance(INDArray v) {
        this.weightsVariance = v;
    }

    public void setWeightsMomentum(INDArray m) {
        this.weightsMomentum = m;
    }

    public void setWeights(INDArray weights) {
        this.weights = weights;
    }

    public void initForAdam() {
        INDArray weightsO = Nd4j.create(this.getWeights().rows(), this.getWeights().columns());
        INDArray biasO = Nd4j.create(this.getBias().rows(), this.getBias().columns());
        this.setWeightsMomentum(weightsO);
        this.setWeightsVariance(weightsO);
        this.setBiasesMomentum(biasO);
        this.setBiasesVariance(biasO);
        Normalization norm = this.getNormalization();
        if (norm != null) {
            INDArray shiftO = Nd4j.create(norm.getShift().rows(), norm.getShift().columns());
            INDArray scaleO = Nd4j.create(norm.getScale().rows(), norm.getScale().columns());
            norm.setShiftMomentum(shiftO);
            norm.setShiftVariance(shiftO);
            norm.setScaleMomentum(scaleO);
            norm.setScaleVariance(scaleO);
        }
    }

    public INDArray getGradientWeights() {
        return gradientWrtWeights;
    }

    public INDArray getGradient() {
        INDArray gradient = null;
        if (this instanceof Output) {
            gradient = ((Output)this).getLoss().gradient(this, ((Output) this).getLabels());
        } else {
            gradient = super.getActFunc().gradient(this, super.getPreActivation());
        }

        return gradient;
    }

    public INDArray gradientWeights(Layer prevLayer, INDArray gradient) {
        INDArray gWrtW = prevLayer.getActivations().transpose().mmul(gradient).div(prevLayer.getActivations().rows());
        return gWrtW;
    }

    public void updateWeights(Optimizer o) {
        this.weights = o.executeWeightsUpdate(this);
    }
    
    public void forwardProp(Layer prev, INDArray data, INDArray labels) {
        Normalization norm = this.getNormalization();
        ActivationFunction actFunc = this.getActFunc();
        MathUtils maths = new MathUtils();
        INDArray z;

        if (prev instanceof Conv2d) {
                INDArray act = prev.getActivations();
                z = act.reshape(act.shape()[0], -1);
        } else if (prev instanceof Dense) {
                z = maths.weightedSum(prev, this);
        } else {
            z = maths.weightedSum(data, this);
        }
            
        this.setPreActivations(z);
        
        // normalize before activation if batch normalization
        if (norm instanceof BatchNormalization) {
            z = norm.normalize(z);
        }

        INDArray activated = actFunc.execute(z);

        // dropout [find a better way to do this]
        if (this.getRegularizers() != null) {
            for (Regularizer r : this.getRegularizers()) {
                if (r instanceof Dropout) {
                    activated = r.regularize(activated);
                }
                break;
            }
        }
            
        this.setActivations(activated);
    }
}