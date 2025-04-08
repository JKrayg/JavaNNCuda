package com.nn.training.loss;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import com.nn.components.Layer;

public class CatCrossEntropy extends Loss {
    public float execute(INDArray activations, INDArray labels) {
        int rows = activations.rows();
        int cols = activations.columns();
        INDArray error = Nd4j.create(rows, cols);
        // need to prevent log(0)
        for (int i = 0; i < rows; i++) {
            error.putRow(i, labels.getRow(i).mul(Transforms.log(activations.getRow(i))));
        }
        return -(error.sumNumber().floatValue() / rows);
    }
    
    // check
    public INDArray gradient(Layer out, INDArray labels) {
        return out.getActivations().sub(labels);
    }
    
}
