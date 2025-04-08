package com.nn.activation;

import org.ejml.simple.SimpleMatrix;

import com.nn.components.Layer;

public class Softmax extends ActivationFunction {
    // weighted sum for single node ∑(wi⋅xi)+b
    public SimpleMatrix execute(SimpleMatrix z) {
        int cols = z.getNumCols();
        int rows = z.getNumRows();
        SimpleMatrix res = new SimpleMatrix(rows, cols);

        for (int j = 0; j < rows; j++) {
            SimpleMatrix currRow = z.getRow(j);
            double max = currRow.elementMax();
            SimpleMatrix expRow = currRow.minus(max).elementExp();
            double sum = expRow.elementSum();
            res.setRow(j, expRow.divide(sum));
        }

        return res;
    }

    public SimpleMatrix derivative(SimpleMatrix z) {
        // ***
        return z;
    }

    public SimpleMatrix gradient(Layer curr, SimpleMatrix gradientWrtPreAct) {
        return gradientWrtPreAct;
        // return gradientWrtPreAct.elementMult(derivative(curr.getPreActivation()));
    }

}
