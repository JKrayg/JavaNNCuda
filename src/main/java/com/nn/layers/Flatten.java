package com.nn.layers;

import com.nn.activation.ActivationFunction;
import com.nn.components.Layer;

public class Flatten extends Layer {

    public Layer initLayer(Layer prev) {
        super.setActivations(prev.getActivations().reshape(1, prev.getActivations().length()));
        return this;
    }
    
}
