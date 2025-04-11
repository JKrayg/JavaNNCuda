package com.nn.components;

import java.util.ArrayList;
import java.util.function.BinaryOperator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import com.nn.activation.ActivationFunction;
import com.nn.activation.Sigmoid;
import com.nn.layers.Output;
import com.nn.training.loss.BinCrossEntropy;
import com.nn.training.loss.Loss;
import com.nn.training.normalization.BatchNormalization;
import com.nn.training.normalization.Normalization;
import com.nn.training.optimizers.Optimizer;
import com.nn.training.regularizers.Regularizer;

public class Layer {
    private int numNeurons;
    private INDArray preActivation;
    private INDArray activations;
    private INDArray weights;
    private INDArray weightsMomentum;
    private INDArray weightsVariance;
    private INDArray bias;
    private INDArray biasMomentum;
    private INDArray biasVariance;
    private INDArray gradientWrtWeights;
    private INDArray gradientWrtBiases;
    private ActivationFunction func;
    private Loss loss;
    private ArrayList<Regularizer> regularizers;
    private Normalization normalization;
    private int inputSize;

    public Layer() {}

    public Layer(int numNeurons, ActivationFunction func, int inputSize) {
        this.numNeurons = numNeurons;
        this.func = func;
        this.inputSize = inputSize;
    }

    public Layer(int numNeurons, ActivationFunction func) {
        this.numNeurons = numNeurons;
        this.func = func;
    }

    public void addRegularizer(Regularizer r) {
        if (regularizers == null) {
            regularizers = new ArrayList<>();
        }
        
        regularizers.add(r);
    }

    public void addNormalization(Normalization n) {
        this.normalization = n;
    }

    public void setWeights(INDArray weights) {
        this.weights = weights;
    }

    public void setBiases(INDArray biases) {
        this.bias = biases;
    }

    public void setWeightsMomentum(INDArray m) {
        this.weightsMomentum = m;
    }

    public void setBiasesMomentum(INDArray m) {
        this.biasMomentum = m;
    }

    public void setWeightsVariance(INDArray v) {
        this.weightsVariance = v;
    }

    public void setBiasesVariance(INDArray v) {
        this.biasVariance = v;
    }

    public void setPreActivations(INDArray preAct) {
        this.preActivation = preAct;
    }

    public void setActivations(INDArray activations) {
        this.activations = activations;
    }

    public void setGradientWeights(INDArray gWrtW) {
        this.gradientWrtWeights = gWrtW;
    }

    public void setGradientBiases(INDArray gWrtB) {
        this.gradientWrtBiases = gWrtB;
    }

    // public void setGradientShift(INDArray gWrtSh) {
    //     ((BatchNormalization) this.normalization).setScale(gWrtSh);
    // }

    // public void setGradientScale(INDArray gWrtSc) {
    //     ((BatchNormalization) this.normalization).setScale(gWrtSc);
    // }

    public void setLoss(Loss loss) {
        this.loss = loss;
    }

    public int getNumNeurons() {
        return numNeurons;
    }

    public INDArray getActivations() {
        return activations;
    }

    public INDArray getPreActivation() {
        return preActivation;
    }

    public INDArray getWeights() {
        return weights;
    }

    public INDArray getBias() {
        return bias;
    }

    public INDArray getWeightsMomentum() {
        return weightsMomentum;
    }

    public INDArray getWeightsVariance() {
        return weightsVariance;
    }

    public INDArray getBiasMomentum() {
        return biasMomentum;
    }

    public INDArray getBiasVariance() {
        return biasVariance;
    }

    public ActivationFunction getActFunc() {
        return func;
    }

    public ArrayList<Regularizer> getRegularizers() {
        return regularizers;
    }

    public Normalization getNormalization() {
        return normalization;
    }

    public int getInputSize() {
        return inputSize;
    }

    public Loss getLoss() {
        return loss;
    }

    public INDArray getGradientWeights() {
        return gradientWrtWeights;
    }

    public INDArray getGradientBias() {
        return gradientWrtBiases;
    }

    public INDArray getGradient() {
        INDArray gradient = null;
        if (this instanceof Output) {
            gradient = this.getLoss().gradient(this, ((Output) this).getLabels());
        } else {
            gradient = func.gradient(this, preActivation);
        }

        return gradient;
    }

    public INDArray gradientWeights(Layer prevLayer, INDArray gradient) {
        INDArray gWrtW = prevLayer.getActivations().transpose().mmul(gradient).div(prevLayer.getActivations().rows());
        return gWrtW;
    }

    public INDArray gradientBias(INDArray gradient) {
        INDArray sums = gradient.sum(0).reshape(gradient.columns(), 1);
        return sums.div(gradient.rows());
        // float[][] biasG = new float[gradient.columns()][1];
        // for (int i = 0; i < gradient.columns(); i++) {
        //     INDArray col = gradient.getColumn(i);
        //     biasG[i][0] = col.sumNumber().floatValue() / gradient.rows();
        // }
        // return Nd4j.create(biasG);
    }

    public void updateWeights(Optimizer o) {
        this.weights = o.executeWeightsUpdate(this);
    }

    public void updateBiases(Optimizer o) {
        this.bias = o.executeBiasUpdate(this);
    }
    
}