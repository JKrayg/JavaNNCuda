package com.nn.activation;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import com.nn.components.Layer;

public class Softmax extends ActivationFunction {
    // weighted sum for single node ∑(wi⋅xi)+b
    public INDArray execute(INDArray z) {
        int cols = z.columns();
        int rows = z.rows();
        INDArray res = Nd4j.create(rows, cols);

        for (int j = 0; j < rows; j++) {
            INDArray currRow = z.getRow(j);
            float max = currRow.maxNumber().floatValue();
            // potential problem -----------------------------------------------------
            INDArray expRow = Transforms.exp(currRow.sub(max));
            float sum = expRow.sumNumber().floatValue();
            res.putRow(j, expRow.div(sum));
        }

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
