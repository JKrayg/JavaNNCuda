package com.nn.layers;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import com.nn.components.Layer;

public class Flatten extends Dense {

    public Layer initLayer(Layer prev, int betachSize) {
        this.setPreActivations(prev.getActivations());
        long[] shape = prev.getActivations().shape();
        INDArray newShape = prev.getActivations().reshape(shape[0], -1);
        this.setActivations(newShape);
        this.setNumNeurons((int)newShape.shape()[1]);
        return this;
    }

    public void forwardProp(Layer prev) {
        this.setPreActivations(prev.getActivations());
        long[] shape = prev.getActivations().shape();
        // System.out.println("shape: " + Arrays.toString(shape));
        INDArray newShape = prev.getActivations().reshape(shape[0], -1);
        // System.out.println("new shape: " + Arrays.toString(newShape.shape()));
        this.setActivations(newShape);
        this.setNumNeurons((int)newShape.shape()[1]);
    }

    public void initForAdam() {}
    
}
