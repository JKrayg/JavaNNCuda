package com.nn.training.normalization;

import org.nd4j.linalg.api.ndarray.INDArray;
import com.nn.training.optimizers.Optimizer;

public abstract class Normalization {
    public abstract boolean isBeforeActivation();
    public abstract void setScale(INDArray scale);
    public abstract void setShift(INDArray shift);
    public abstract void setMeans(INDArray means);
    public abstract void setVariances(INDArray variances);
    public abstract void setShiftMomentum(INDArray shM);
    public abstract void setShiftVariance(INDArray shV);
    public abstract void setScaleMomentum(INDArray scM);
    public abstract void setScaleVariance(INDArray scV);
    public abstract void updateScale(Optimizer o);
    public abstract void updateShift(Optimizer o);
    public abstract INDArray getShiftMomentum();
    public abstract INDArray getShiftVariance();
    public abstract INDArray getScaleMomentum();
    public abstract INDArray getScaleVariance();
    public abstract INDArray getScale();
    public abstract INDArray getShift();
    public abstract INDArray getGradientShift();
    public abstract INDArray getGradientScale();
    public abstract INDArray gradientShift(INDArray gradient);
    public abstract INDArray gradientScale(INDArray gradient);
    public abstract INDArray getNormZ();
    public abstract INDArray normalize(INDArray z);
    
}
