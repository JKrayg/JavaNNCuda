package com.nn.training.normalization;

import org.ejml.simple.SimpleMatrix;

import com.nn.training.optimizers.Optimizer;

public class BatchNormalization extends Normalization {
    private SimpleMatrix scale;
    private SimpleMatrix shift;
    private SimpleMatrix means;
    private SimpleMatrix variances;
    private SimpleMatrix runningMeans;
    private SimpleMatrix runningVariances;
    private SimpleMatrix shiftMomentum;
    private SimpleMatrix shiftVariance;
    private SimpleMatrix scaleMomentum;
    private SimpleMatrix scaleVariance;
    private double momentum = 0.99;
    private double epsilon = 1e-3;
    private boolean beforeActivation = true;
    private SimpleMatrix gradientWrtShift;
    private SimpleMatrix gradientWrtScale;
    private SimpleMatrix preNormZ;
    private SimpleMatrix preScaleShiftZ;
    private SimpleMatrix normalizedZ;

    public BatchNormalization() {
    }

    public void setScale(SimpleMatrix scale) {
        this.scale = scale;
    }

    public void setShift(SimpleMatrix shift) {
        this.shift = shift;
    }

    public void setMeans(SimpleMatrix means) {
        this.means = means;
    }

    public void setRunningMeans(SimpleMatrix means) {
        this.runningMeans = means;
    }

    public void setVariances(SimpleMatrix variances) {
        this.variances = variances;
    }
    

    public void setRunningVariances(SimpleMatrix variances) {
        this.runningVariances = variances;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public void setShiftMomentum(SimpleMatrix shM) {
        this.shiftMomentum = shM;
    }

    public void setShiftVariance(SimpleMatrix shV) {
        this.shiftVariance = shV;
    }

    public void setScaleMomentum(SimpleMatrix scM) {
        this.scaleMomentum = scM;
    }

    public void setScaleVariance(SimpleMatrix scV) {
        this.scaleVariance = scV;
    }

    public void setGradientShift(SimpleMatrix gWrtSh) {
        this.gradientWrtShift = gWrtSh;
    }

    public void setGradientScale(SimpleMatrix gWrtSc) {
        this.gradientWrtScale = gWrtSc;
    }

    public void beforeActivation(boolean b) {
        this.beforeActivation = b;
    }

    public boolean isBeforeActivation() {
        return beforeActivation;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public SimpleMatrix getScale() {
        return scale;
    }

    public SimpleMatrix getShift() {
        return shift;
    }

    public SimpleMatrix getRunningMeans() {
        return runningMeans;
    }

    public SimpleMatrix getRunningVariances() {
        return runningVariances;
    }

    public SimpleMatrix getMeans() {
        return means;
    }

    public SimpleMatrix getVariances() {
        return variances;
    }

    public double getMomentum() {
        return momentum;
    }

    public SimpleMatrix getShiftMomentum() {
        return shiftMomentum;
    }

    public SimpleMatrix getShiftVariance() {
        return shiftVariance;
    }

    public SimpleMatrix getScaleMomentum() {
        return scaleMomentum;
    }

    public SimpleMatrix getScaleVariance() {
        return scaleVariance;
    }

    public SimpleMatrix getGradientShift() {
        return gradientWrtShift;
    }

    public SimpleMatrix getGradientScale() {
        return gradientWrtScale;
    }

    public SimpleMatrix getNormZ() {
        return normalizedZ;
    }

    public SimpleMatrix getPreScaleShiftZ() {
        return preScaleShiftZ;
    }

    public SimpleMatrix getPreNormZ() {
        return preNormZ;
    }

    public SimpleMatrix gradientShift(SimpleMatrix gradient) {
        // System.out.println("dL/dzHat sum: " + gradient.elementSum());
        int cols = gradient.getNumCols();
        int rows = gradient.getNumRows();
        SimpleMatrix gWrtSh = new SimpleMatrix(cols, 1);

        for (int i = 0; i < cols; i++) {
            gWrtSh.set(i, 0, gradient.getColumn(i).elementSum());
        }

        return gWrtSh;
    }

    public SimpleMatrix gradientScale(SimpleMatrix gradient) {
        // System.out.println("sumsc: " + gradient.elementSum());
        SimpleMatrix gWrtSc = new SimpleMatrix(gradient.getNumCols(), 1);
        int cols = gradient.getNumCols();
        int rows = gradient.getNumRows();

        for (int i = 0; i < cols; i++) {
            gWrtSc.set(i, 0, gradient.getColumn(i).elementMult(preScaleShiftZ.getColumn(i)).elementSum());
        }
        return gWrtSc;
    }

    public void updateScale(Optimizer o) {
        this.setScale(o.executeScaleUpdate(this));
    }

    public void updateShift(Optimizer o) {
        this.setShift(o.executeShiftUpdate(this));
    }

    public SimpleMatrix means(SimpleMatrix z) {
        int rows = z.getNumRows();
        int cols = z.getNumCols();
        SimpleMatrix means = new SimpleMatrix(cols, 1);

        for (int i = 0; i < cols; i++) {
            means.set(i, z.getColumn(i).elementSum() / rows);
        }

        return means;
    }

    public SimpleMatrix variances(SimpleMatrix z) {
        int rows = z.getNumRows();
        int cols = z.getNumCols();
        SimpleMatrix variances = new SimpleMatrix(cols, 1);
        SimpleMatrix means = means(z);

        for (int i = 0; i < cols; i++) {
            variances.set(i, z.getColumn(i).minus(means.get(i)).elementPower(2).elementSum() / rows);
        }

        return variances;
    }

    public SimpleMatrix normalize(SimpleMatrix z) {
        int rows = z.getNumRows();
        int cols = z.getNumCols();
        SimpleMatrix means = means(z);
        SimpleMatrix variances = variances(z);
        SimpleMatrix preSclShft = new SimpleMatrix(rows, cols);
        SimpleMatrix norm = new SimpleMatrix(rows, cols);

        for (int i = 0; i < cols; i++) {
            SimpleMatrix currCol = z.getColumn(i);
            SimpleMatrix normalizedCol = currCol.minus(means.get(i)).divide(Math.sqrt(variances.get(i) + epsilon));
            preSclShft.setColumn(i, normalizedCol);
            norm.setColumn(i, normalizedCol.scale(scale.get(i)).plus(shift.get(i)));
        }

        this.means = means;
        this.variances = variances;
        this.runningMeans = runningMeans.scale(momentum).plus(means.scale((1 - momentum)));
        this.runningVariances = runningVariances.scale(momentum).plus(variances.scale((1 - momentum)));
        this.preNormZ = z.copy();
        this.preScaleShiftZ = preSclShft;

        return norm;
    }

    public void setPreNormZ(SimpleMatrix z) {
        this.preNormZ = z;
    }


    public SimpleMatrix gradientPreBN(SimpleMatrix dLdzHat) {
        int rows = dLdzHat.getNumRows();
        int cols = dLdzHat.getNumCols();
        SimpleMatrix dLdz = new SimpleMatrix(rows, cols);
        SimpleMatrix std = variances.plus(epsilon).elementPower(0.5);
        SimpleMatrix scalingFactor = scale.elementDiv(std);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double gradientElement = dLdzHat.get(i, j);
                double normElement = preScaleShiftZ.get(i, j);
                dLdz.set(i, j,
                    scalingFactor.get(i) * (gradientElement - 
                    ((gradientElement - normElement) / rows) * 
                    ((gradientElement * normElement) / rows)));
            }
            
        }

        return dLdz;
    }

    public SimpleMatrix gradientPreBNSimple(SimpleMatrix dLdzHat) {
        int rows = dLdzHat.getNumRows();
        int cols = dLdzHat.getNumCols();
        SimpleMatrix dLdz = new SimpleMatrix(rows, cols);
        SimpleMatrix std = variances.plus(epsilon).elementPower(0.5);
        SimpleMatrix scalingFactor = scale.elementDiv(std);

        for (int i = 0; i < rows; i++) {
            dLdz.setRow(i, dLdzHat.getRow(i).transpose().elementMult(scalingFactor));
        }

        return dLdz;
    }
}
