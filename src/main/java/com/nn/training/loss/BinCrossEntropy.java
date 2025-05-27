package com.nn.training.loss;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import com.nn.components.Layer;

public class BinCrossEntropy extends Loss {
    public float execute(INDArray activations, INDArray labels) {
        // System.out.println("a: " + Arrays.toString(activations.shape()));
        // System.out.println("l: " + Arrays.toString(labels.shape()));
        float epsilon = 1e-7f;
        INDArray clipped = Transforms.max(activations, Nd4j.scalar(epsilon));
        clipped = Transforms.min(clipped, Nd4j.scalar(1.0f - epsilon));

        INDArray a = labels.mul(Transforms.log(clipped));
        INDArray c = Transforms.log(Nd4j.ones(clipped.shape()).subi(clipped));
        INDArray d = Nd4j.ones(labels.shape()).subi(labels).muli(c);

        return a.addi(d).muli(-1).divi(labels.rows()).sumNumber().floatValue();
    }

    public INDArray gradient(Layer out, INDArray labels) {
        return out.getActivations().sub(labels);
    }
}
