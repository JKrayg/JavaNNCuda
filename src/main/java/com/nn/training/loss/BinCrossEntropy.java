package com.nn.training.loss;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import com.nn.components.Layer;

public class BinCrossEntropy extends Loss {
    public float execute(INDArray activations, INDArray labels) {
        float sumLoss = 0.0f;
        float n = activations.length();
        INDArray pred1 = Transforms.log(activations);
        INDArray firstPart = labels.mul(pred1);
        INDArray pred2 = Transforms.log(Nd4j.ones(activations.rows()).sub(activations));
        INDArray secondPart = Nd4j.ones(labels.rows()).sub(labels).mul(pred2);
        return firstPart.mul(secondPart).mul(-1).sumNumber().floatValue();

        // for (int i = 0; i < n; i++) {
        //     float pred = activations.getFloat(i);
        //     float y = labels.getFloat(i);
        //     float epsilon = (float) 1e-8;
        //     // need to prevent log(0)
        //     pred = Math.max(epsilon, Math.min(1 - epsilon, pred));
        //     sumLoss += -(y * Math.log(pred) + (1 - y) * Math.log(1 - pred));
        // }
        // return sumLoss / n;
    }

    public INDArray gradient(Layer out, INDArray labels) {
        return out.getActivations().sub(labels);
    }
}
