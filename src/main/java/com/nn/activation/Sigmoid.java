package com.nn.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Exp;
import org.nd4j.linalg.factory.Nd4j;

public class Sigmoid extends ActivationFunction {
    public INDArray execute(INDArray z) {
        INDArray dup = z.dup();
        Nd4j.getExecutioner().exec(new Exp(dup.mul(-1))).addi(1).divi(1);
        // int rows = z.rows();
        // float[] v = new float[z.rows()];

        // for (int i = 0; i < rows; i++) {
        //     v[i] = (float) (1 / (1 + Math.exp(-(z.getFloat(i)))));
        // }

        // INDArray out = Nd4j.create(v);
        return dup;
    }

    public INDArray derivative(INDArray z) {
        // ***
        return z;
    }

    public INDArray gradient(INDArray preactivation, INDArray gradientWrtPreAct) {
        return gradientWrtPreAct.mul(derivative(preactivation));
    }
}
