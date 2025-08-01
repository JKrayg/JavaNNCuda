package com.nn.training.loss;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.nn.components.Layer;

public class PPO extends Loss{
    
    public INDArray ppoExecute(INDArray surrogate) {
        return surrogate.neg().mean();
    }

    public INDArray ppoGradient(INDArray advantage, INDArray ratio) {
        return Nd4j.createUninitialized(0);
        
    }

    @Override
    public INDArray gradient(Layer out, INDArray preds) {
        return Nd4j.createUninitialized(0);
        
    }

    @Override
    public float execute(INDArray activations, INDArray preds) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'execute'");
    }
    
}
