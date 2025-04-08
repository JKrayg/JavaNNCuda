package com.nn.training.loss;

import org.ejml.simple.SimpleMatrix;
import com.nn.components.Layer;

public class CatCrossEntropy extends Loss {
    public double execute(SimpleMatrix activations, SimpleMatrix labels) {
        int rows = activations.getNumRows();
        int cols = activations.getNumCols();
        SimpleMatrix error = new SimpleMatrix(rows, cols);
        // need to prevent log(0)
        for (int i = 0; i < rows; i++) {
            error.setRow(i, labels.getRow(i).elementMult(activations.getRow(i).elementLog()));
        }
        return -(error.elementSum() / rows);
    }
    
    // check
    public SimpleMatrix gradient(Layer out, SimpleMatrix labels) {
        return out.getActivations().minus(labels);
    }
    
}
