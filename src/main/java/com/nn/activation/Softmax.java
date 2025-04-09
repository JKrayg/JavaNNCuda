package com.nn.activation;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import com.nn.components.Layer;

public class Softmax extends ActivationFunction {
    // weighted sum for single node ∑(wi⋅xi)+b
    public INDArray execute(INDArray z) {
        INDArray max = z.max(1);
        INDArray zs = z.subColumnVector(max);
        INDArray expZ = Transforms.exp(zs);
        INDArray sumExpZ = expZ.sum(1);
        INDArray res = expZ.divColumnVector(sumExpZ);

        return res;
    }

    public INDArray derivative(INDArray z) {
        // ***
        return z;
    }

    public INDArray gradient(Layer curr, INDArray gradientWrtPreAct) {
        return gradientWrtPreAct;
        // return gradientWrtPreAct.elementMult(derivative(curr.getPreActivation()));
    }

}
