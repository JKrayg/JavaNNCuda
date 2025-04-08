package com.nn.training.loss;

import org.nd4j.linalg.api.ndarray.INDArray;
import com.nn.components.Layer;

public class SparseCrossEntropy extends Loss {
    public float execute(INDArray activations, INDArray labels) {
        return 0.0f;
    }

    public INDArray gradient(Layer out, INDArray labels) {
        // ***
        return out.getActivations().sub(labels);
    }
}
