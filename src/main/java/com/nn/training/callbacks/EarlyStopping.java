package com.nn.training.callbacks;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class EarlyStopping extends Callback {
    private String monitor;
    private float minChange = 0;
    private int patience = 0;
    private float low = 0.0f;
    private int patidx = 0;

    public EarlyStopping(String monitor, double minChange, int patience) {
        this.monitor = monitor;
        this.minChange = (float) minChange;
        this.patience = patience;
    }

    public String getMetric() {
        return monitor;
    }

    public float getMinChange() {
        return minChange;
    }

    public int getPatience() {
        return patience;
    }

    // fix this
    public boolean checkStop(float d) {
        if (low == 0.0f) {
            low = d;
        } else {
            if (monitor.equals("accuracy") || monitor.equals("val_accuracy")) {
                if (d < low + minChange) {
                    patidx++;
                    if (patidx == patience - 1) {
                        return true;
                    }
                } else {
                    low = d;
                    patidx = 0;
                }

            } else if (monitor.equals("val_loss") || monitor.equals("loss")) {
                if (d > low - minChange) {
                    patidx++;
                    if (patidx == patience - 1) {
                        return true;
                    }
                } else {
                    low = d;
                    patidx = 0;
                }
            }

        }

        // System.out.println("new low: " + low + "\n");

        return false;
    }

}
