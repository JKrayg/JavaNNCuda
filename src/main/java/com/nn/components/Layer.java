package com.nn.components;

import java.util.ArrayList;
import java.util.Arrays;
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
    // private int numNeurons;
    private Layer next;
    private Layer previous;
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
    // private Loss loss;
    private ArrayList<Regularizer> regularizers;
    private Normalization normalization;
    private int numFeatures;

    public Layer() {}

    public Layer(ActivationFunction func, int numFeatures) {
        this.func = func;
        this.numFeatures = numFeatures;
    }

    public Layer(ActivationFunction func) {
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

    // public void setWeights(INDArray weights) {
    //     this.weights = weights;
    // }

    public void setNext(Layer next) {
        this.next = next;
    }

    public void setPrev(Layer prev) {
        this.previous = prev;
    }

    public void setActivationFunction(ActivationFunction a) {
        this.func = a;
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

    // public void setLoss(Loss loss) {
    //     this.loss = loss;
    // }

    // public int getNumNeurons() {
    //     return numNeurons;
    // }

    public Layer getNext() {
        return next;
    }

    public Layer getPrev() {
        return previous;
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

    public int getNumFeatures() {
        return numFeatures;
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

    // public int getNumFeatures() {
    //     return numFeatures;
    // }

    // // public Loss getLoss() {
    // //     return loss;
    // // }

    public INDArray getGradientWeights() {
        return gradientWrtWeights;
    }

    public INDArray getGradientBias() {
        return gradientWrtBiases;
    }

    public INDArray getGradient() {
        INDArray gradient = null;
        if (this instanceof Output) {
            gradient = ((Output)this).getLoss().gradient(this, ((Output) this).getLabels());
        } else {
            gradient = func.gradient(this.getPreActivation(), preActivation);
        }

        return gradient;
    }

    public INDArray gradientWeights(INDArray activation, INDArray gradient) {
        INDArray gWrtW = activation.transpose().mmul(gradient).div(activation.rows());
        return gWrtW;
    }

    public INDArray gradientBias(INDArray gradient) {
        INDArray sums = gradient.sum(0).reshape(gradient.columns(), 1);
        return sums.div(gradient.rows());
    }

    public void updateWeights(Optimizer o) {}

    public void updateBiases(Optimizer o) {
        this.bias = o.executeBiasUpdate(this);
    }

    public void initForAdam() {}

    public Layer initLayer(Layer prev, int batchSize) {
        return new Layer();
    }

    public void forwardProp(Layer prev) {}

    public void getGradients(Layer prev, INDArray gradient, INDArray data) {
        System.out.println("this is called");
    }

    public String toString() {
        String s = "";
        s += "class: " + this.getClass().getSimpleName() + "\n";
        s += "preactivation: " + Arrays.toString(this.getPreActivation().shape()) + "\n";
        s += "activations: " + Arrays.toString(this.getActivations().shape()) + "\n";
        s += "weights: " + Arrays.toString(this.getWeights().shape()) + "\n";
        s += "bias: " + Arrays.toString(this.getBias().shape()) + "\n";
        s += "weights momentum: " + Arrays.toString(this.getWeightsMomentum().shape()) + "\n";
        s += "weights variance: " + Arrays.toString(this.getWeightsVariance().shape()) + "\n";
        s += "bias momentum: " + Arrays.toString(this.getBiasMomentum().shape()) + "\n";
        s += "bias variance: " + Arrays.toString(this.getBiasVariance().shape()) + "\n";
        s += "gradient wrt weights: " + Arrays.toString(this.getGradientWeights().shape()) + "\n";
        s += "gradient wrt bias: " + Arrays.toString(this.getGradientBias().shape());

        return s;
    }

}