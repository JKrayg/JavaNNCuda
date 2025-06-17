package com.nn.layers;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import com.nn.activation.ActivationFunction;
import com.nn.components.Layer;
import com.nn.training.loss.Loss;
import com.nn.training.regularizers.Regularizer;

public class Output extends Dense {
    private INDArray preds;
    private Loss loss;

    public Output(int numNeurons) {
        super(numNeurons);
    }

    public Output(int numNeurons, ActivationFunction actFunc) {
        super(numNeurons, actFunc);
    }

    public Output(int numNeurons, ActivationFunction actFunc, Loss loss) {
        super(numNeurons, actFunc);
        this.loss = loss;
    }

    public Loss getLoss() {
        return loss;
    }

    public void setPreds(INDArray preds) {
        this.preds = preds;
    }

    public INDArray getPreds() {
        return preds;
    }

    // check
    public INDArray gradientWeights(Layer prevLayer, INDArray gradientWrtOutput) {
        return prevLayer.getActivations().transpose()
                .mmul(gradientWrtOutput).div(preds.length());
    }

    // check
    public INDArray gradientBias(INDArray gradientWrtOutput) {
        INDArray sums = gradientWrtOutput.sum(0)
                .reshape(gradientWrtOutput.columns(), 1);
                
        return sums.div(preds.length());
    }
}
