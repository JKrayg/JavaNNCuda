package com.nn.layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import com.nn.activation.ActivationFunction;
import com.nn.components.Layer;
import com.nn.training.loss.Loss;
import com.nn.training.regularizers.Regularizer;

public class Output extends Layer {
    private INDArray labels;
    private Loss loss;

    public Output(int numNeurons, ActivationFunction actFunc, Loss loss) {
        super(numNeurons, actFunc);
        this.loss = loss;
    }

    public Loss getLoss() {
        return loss;
    }

    public void setLabels(INDArray labels) {
        this.labels = labels;
    }

    public INDArray getLabels() {
        return labels;
    }

    // check
    public INDArray gradientWeights(Layer prevLayer, INDArray gradientWrtOutput) {
        return prevLayer.getActivations().transpose().mmul(gradientWrtOutput).div(labels.length());
    }

    // check
    public INDArray gradientBias(INDArray gradientWrtOutput) {
        float[][] biasG = new float[this.getNumNeurons()][1];
        for (int i = 0; i < gradientWrtOutput.columns(); i++) {
            INDArray col = gradientWrtOutput.getColumn(i);
            biasG[i][0] = col.sumNumber().floatValue() / labels.length();
        }
        return Nd4j.create(biasG);
    }
}
