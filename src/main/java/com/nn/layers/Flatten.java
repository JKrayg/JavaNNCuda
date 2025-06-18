package com.nn.layers;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import com.nn.components.Layer;

public class Flatten extends Dense {

    private long[] prevShape;

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
        this.prevShape = shape;
        INDArray newShape = prev.getActivations().reshape(shape[0], -1);
        this.setActivations(newShape);
        this.setNumNeurons((int)newShape.shape()[1]);
    }

    public INDArray reshapeGradient(INDArray gradient) {
        return gradient.reshape(prevShape);
    }

    public void getGradients(Layer prev, INDArray gradient, INDArray data) {
        System.out.println("incoming gradient from flatten: " + Arrays.toString(gradient.shape()));
        prev.getGradients(prev.getPrev(), gradient, data);
    }

    public void initForAdam() {}

    public String toString() {
        return "class: " + this.getClass().getSimpleName() + "\n" +
               "prev activations: " + Arrays.toString(this.prevShape) + "\n" +
               "activations: " + Arrays.toString(this.getActivations().shape());
    }
    
}
