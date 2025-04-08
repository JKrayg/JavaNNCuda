package com.nn.layers;

import org.ejml.simple.SimpleMatrix;
import com.nn.activation.ActivationFunction;
import com.nn.components.Layer;
import com.nn.training.loss.Loss;
import com.nn.training.regularizers.Regularizer;

public class Output extends Layer {
    private SimpleMatrix labels;
    private Loss loss;

    public Output(int numNeurons, ActivationFunction actFunc, Loss loss) {
        super(numNeurons, actFunc);
        this.loss = loss;
    }

    public Loss getLoss() {
        return loss;
    }

    public void setLabels(SimpleMatrix labels) {
        this.labels = labels;
    }

    public SimpleMatrix getLabels() {
        return labels;
    }

    // check
    public SimpleMatrix gradientWeights(Layer prevLayer, SimpleMatrix gradientWrtOutput) {
        return prevLayer.getActivations().transpose().mult(gradientWrtOutput).divide(labels.getNumElements());
    }

    // check
    public SimpleMatrix gradientBias(SimpleMatrix gradientWrtOutput) {
        double[] biasG = new double[this.getNumNeurons()];
        for (int i = 0; i < gradientWrtOutput.getNumCols(); i++) {
            SimpleMatrix col = gradientWrtOutput.extractVector(false, i);
            biasG[i] = col.elementSum() / labels.getNumElements();
        }
        return new SimpleMatrix(biasG);
    }
}
