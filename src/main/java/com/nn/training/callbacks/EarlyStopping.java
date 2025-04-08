package com.nn.training.callbacks;

public class EarlyStopping extends Callback{
    private String monitor;
    private double minChange = 0;
    private int patience = 0;
    private int epochStart = 0;

    public EarlyStopping(String monitor) {
        this.monitor = monitor;
    }

    public void setMinChange(int minChange) {
        this.minChange = minChange;
    }
    public void setPatience(int patience) {
        this.patience = patience;
    }
    public void setEpochStart(int epochStart) {
        this.epochStart = epochStart;
    }

    public String getMonitor() {
        return monitor;
    }

    public double getMinChange() {
        return minChange;
    }

    public int getPatience() {
        return patience;
    }

    public int getEpochStart() {
        return epochStart;
    }

    // public boolean checkStop(double[] lossHistory) {

    // }
}
