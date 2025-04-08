package com.nn.training.normalization;

import org.ejml.simple.SimpleMatrix;

import com.nn.training.optimizers.Optimizer;

public abstract class Normalization {
    public abstract boolean isBeforeActivation();
    public abstract void setScale(SimpleMatrix scale);
    public abstract void setShift(SimpleMatrix shift);
    public abstract void setMeans(SimpleMatrix means);
    public abstract void setVariances(SimpleMatrix variances);
    public abstract void setShiftMomentum(SimpleMatrix shM);
    public abstract void setShiftVariance(SimpleMatrix shV);
    public abstract void setScaleMomentum(SimpleMatrix scM);
    public abstract void setScaleVariance(SimpleMatrix scV);
    public abstract void updateScale(Optimizer o);
    public abstract void updateShift(Optimizer o);
    public abstract SimpleMatrix getShiftMomentum();
    public abstract SimpleMatrix getShiftVariance();
    public abstract SimpleMatrix getScaleMomentum();
    public abstract SimpleMatrix getScaleVariance();
    public abstract SimpleMatrix getScale();
    public abstract SimpleMatrix getShift();
    public abstract SimpleMatrix getGradientShift();
    public abstract SimpleMatrix getGradientScale();
    public abstract SimpleMatrix gradientShift(SimpleMatrix gradient);
    public abstract SimpleMatrix gradientScale(SimpleMatrix gradient);
    public abstract SimpleMatrix getNormZ();
    public abstract SimpleMatrix normalize(SimpleMatrix z);
    
}
