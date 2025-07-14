package com.nn.training.loss;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.nn.components.Layer;

public class PPO extends Loss{

    @Override
    public float execute(INDArray activations, INDArray preds) {
        return 0.0f;
    }

    @Override
    public INDArray gradient(Layer out, INDArray preds) {
        return Nd4j.createUninitialized(0);
        
    }
    
}
