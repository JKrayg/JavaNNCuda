package com.nn.components;

import java.util.ArrayList;
import java.util.function.BinaryOperator;

import org.ejml.simple.SimpleMatrix;
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
    private SimpleMatrix preActivation;
    private SimpleMatrix activations;
    private SimpleMatrix weights;
    private SimpleMatrix weightsMomentum;
    private SimpleMatrix weightsVariance;
    private SimpleMatrix bias;
    private SimpleMatrix biasMomentum;
    private SimpleMatrix biasVariance;
    private SimpleMatrix gradientWrtWeights;
    private SimpleMatrix gradientWrtBiases;
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

    public void setWeights(SimpleMatrix weights) {
        this.weights = weights;
    }

    public void setBiases(SimpleMatrix biases) {
        this.bias = biases;
    }

    public void setWeightsMomentum(SimpleMatrix m) {
        this.weightsMomentum = m;
    }

    public void setBiasesMomentum(SimpleMatrix m) {
        this.biasMomentum = m;
    }

    public void setWeightsVariance(SimpleMatrix v) {
        this.weightsVariance = v;
    }

    public void setBiasesVariance(SimpleMatrix v) {
        this.biasVariance = v;
    }

    public void setPreActivations(SimpleMatrix preAct) {
        this.preActivation = preAct;
    }

    public void setActivations(SimpleMatrix activations) {
        this.activations = activations;
    }

    public void setGradientWeights(SimpleMatrix gWrtW) {
        this.gradientWrtWeights = gWrtW;
    }

    public void setGradientBiases(SimpleMatrix gWrtB) {
        this.gradientWrtBiases = gWrtB;
    }

    // public void setGradientShift(SimpleMatrix gWrtSh) {
    //     ((BatchNormalization) this.normalization).setScale(gWrtSh);
    // }

    // public void setGradientScale(SimpleMatrix gWrtSc) {
    //     ((BatchNormalization) this.normalization).setScale(gWrtSc);
    // }

    public void setLoss(Loss loss) {
        this.loss = loss;
    }

    public int getNumNeurons() {
        return numNeurons;
    }

    public SimpleMatrix getActivations() {
        return activations;
    }

    public SimpleMatrix getPreActivation() {
        return preActivation;
    }

    public SimpleMatrix getWeights() {
        return weights;
    }

    public SimpleMatrix getBias() {
        return bias;
    }

    public SimpleMatrix getWeightsMomentum() {
        return weightsMomentum;
    }

    public SimpleMatrix getWeightsVariance() {
        return weightsVariance;
    }

    public SimpleMatrix getBiasMomentum() {
        return biasMomentum;
    }

    public SimpleMatrix getBiasVariance() {
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

    public SimpleMatrix getGradientWeights() {
        return gradientWrtWeights;
    }

    public SimpleMatrix getGradientBias() {
        return gradientWrtBiases;
    }

    public SimpleMatrix getGradient() {
        SimpleMatrix gradient = null;
        if (this instanceof Output) {
            gradient = this.getLoss().gradient(this, ((Output) this).getLabels());
        } else {
            gradient = func.gradient(this, preActivation);
        }

        return gradient;
    }

    public SimpleMatrix gradientWeights(Layer prevLayer, SimpleMatrix gradient) {
        SimpleMatrix gWrtW = prevLayer.getActivations().transpose().mult(gradient).divide(prevLayer.getActivations().getNumRows());
        return gWrtW;
    }

    public SimpleMatrix gradientBias(SimpleMatrix gradient) {
        double[] biasG = new double[gradient.getNumCols()];
        for (int i = 0; i < gradient.getNumCols(); i++) {
            SimpleMatrix col = gradient.extractVector(false, i);
            biasG[i] = col.elementSum() / gradient.getNumRows();
        }
        return new SimpleMatrix(biasG);
    }

    public void updateWeights(Optimizer o) {
        this.weights = o.executeWeightsUpdate(this);
    }

    public void updateBiases(Optimizer o) {
        this.bias = o.executeBiasUpdate(this);
    }
    
}