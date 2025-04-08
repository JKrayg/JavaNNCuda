package com.nn.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import com.nn.components.Layer;

public class ReLU extends ActivationFunction {
    public INDArray execute(INDArray z) {
        int rows = z.rows();
        int cols = z.columns();
        INDArray ez = Nd4j.create(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float curr = z.getFloat(i, j);
                ez.putScalar(i, j, curr > 0 ? curr : 0);
            }
            
        }
        return ez;
    }

    public INDArray derivative(INDArray z) {
        int rows = z.rows();
        int cols = z.columns();
        INDArray dz = Nd4j.create(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                dz.putScalar(i, j, z.getFloat(i, j) > 0 ? 1.0 : 0.0);
            }
            
        }
        return dz;
    }

    public INDArray gradient(Layer prev, INDArray gradientWrtPreAct) {
        return derivative(prev.getPreActivation()).mul(gradientWrtPreAct);
    }
}
