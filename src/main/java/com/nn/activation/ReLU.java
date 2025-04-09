package com.nn.activation;


import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

import com.nn.components.Layer;

public class ReLU extends ActivationFunction {
    public INDArray execute(INDArray z) {
        INDArray d = z.dup();
        INDArray az = d.gt(0).castTo(DataType.FLOAT).mul(z);
        return az;
    }

    public INDArray derivative(INDArray z) {
        INDArray d = z.dup();
        INDArray dz = d.gt(0).castTo(DataType.FLOAT);
        return dz;
    }

    public INDArray gradient(Layer prev, INDArray gradientWrtPreAct) {
        return derivative(prev.getPreActivation()).mul(gradientWrtPreAct);
    }
}
