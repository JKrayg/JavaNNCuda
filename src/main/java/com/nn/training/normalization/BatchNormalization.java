package com.nn.training.normalization;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import com.nn.training.optimizers.Optimizer;

public class BatchNormalization extends Normalization {
    private INDArray scale;
    private INDArray shift;
    private INDArray means;
    private INDArray variances;
    private INDArray runningMeans;
    private INDArray runningVariances;
    private INDArray shiftMomentum;
    private INDArray shiftVariance;
    private INDArray scaleMomentum;
    private INDArray scaleVariance;
    private float momentum = 0.99f;
    private float epsilon = 1e-3f;
    private boolean beforeActivation = true;
    private INDArray gradientWrtShift;
    private INDArray gradientWrtScale;
    private INDArray preNormZ;
    private INDArray preScaleShiftZ;
    private INDArray normalizedZ;

    public BatchNormalization() {
    }

    public void setScale(INDArray scale) {
        this.scale = scale;
    }

    public void setShift(INDArray shift) {
        this.shift = shift;
    }

    public void setMeans(INDArray means) {
        this.means = means;
    }

    public void setRunningMeans(INDArray means) {
        this.runningMeans = means;
    }

    public void setVariances(INDArray variances) {
        this.variances = variances;
    }
    

    public void setRunningVariances(INDArray variances) {
        this.runningVariances = variances;
    }

    public void setMomentum(float momentum) {
        this.momentum = momentum;
    }

    public void setEpsilon(float epsilon) {
        this.epsilon = epsilon;
    }

    public void setShiftMomentum(INDArray shM) {
        this.shiftMomentum = shM;
    }

    public void setShiftVariance(INDArray shV) {
        this.shiftVariance = shV;
    }

    public void setScaleMomentum(INDArray scM) {
        this.scaleMomentum = scM;
    }

    public void setScaleVariance(INDArray scV) {
        this.scaleVariance = scV;
    }

    public void setGradientShift(INDArray gWrtSh) {
        this.gradientWrtShift = gWrtSh;
    }

    public void setGradientScale(INDArray gWrtSc) {
        this.gradientWrtScale = gWrtSc;
    }

    public void beforeActivation(boolean b) {
        this.beforeActivation = b;
    }

    public boolean isBeforeActivation() {
        return beforeActivation;
    }

    public float getEpsilon() {
        return epsilon;
    }

    public INDArray getScale() {
        return scale;
    }

    public INDArray getShift() {
        return shift;
    }

    public INDArray getRunningMeans() {
        return runningMeans;
    }

    public INDArray getRunningVariances() {
        return runningVariances;
    }

    public INDArray getMeans() {
        return means;
    }

    public INDArray getVariances() {
        return variances;
    }

    public float getMomentum() {
        return momentum;
    }

    public INDArray getShiftMomentum() {
        return shiftMomentum;
    }

    public INDArray getShiftVariance() {
        return shiftVariance;
    }

    public INDArray getScaleMomentum() {
        return scaleMomentum;
    }

    public INDArray getScaleVariance() {
        return scaleVariance;
    }

    public INDArray getGradientShift() {
        return gradientWrtShift;
    }

    public INDArray getGradientScale() {
        return gradientWrtScale;
    }

    public INDArray getNormZ() {
        return normalizedZ;
    }

    public INDArray getPreScaleShiftZ() {
        return preScaleShiftZ;
    }

    public INDArray getPreNormZ() {
        return preNormZ;
    }

    public INDArray gradientShift(INDArray gradient) {
        // System.out.println("dL/dzHat sum: " + gradient.elementSum());
        int cols = gradient.columns();
        int rows = gradient.rows();
        INDArray gWrtSh = Nd4j.create(cols, 1);

        for (int i = 0; i < cols; i++) {
            gWrtSh.put(i, 0, gradient.getColumn(i).sumNumber().floatValue());
        }

        return gWrtSh;
    }

    public INDArray gradientScale(INDArray gradient) {
        // System.out.println("sumsc: " + gradient.elementSum());
        int cols = gradient.columns();
        int rows = gradient.rows();
        INDArray gWrtSc = Nd4j.create(cols, 1);

        for (int i = 0; i < cols; i++) {
            gWrtSc.put(i, 0, gradient.getColumn(i).mul(preScaleShiftZ.getColumn(i)).sumNumber().floatValue());
        }
        return gWrtSc;
    }

    public void updateScale(Optimizer o) {
        this.setScale(o.executeScaleUpdate(this));
    }

    public void updateShift(Optimizer o) {
        this.setShift(o.executeShiftUpdate(this));
    }

    public INDArray means(INDArray z) {
        int rows = z.rows();
        int cols = z.columns();
        INDArray means = Nd4j.create(cols, 1);

        for (int i = 0; i < cols; i++) {
            means.putScalar(i, z.getColumn(i).sumNumber().floatValue() / rows);
        }

        return means;
    }

    public INDArray variances(INDArray z) {
        int rows = z.rows();
        int cols = z.columns();
        INDArray variances = Nd4j.create(cols, 1);
        INDArray means = means(z);

        for (int i = 0; i < cols; i++) {
            INDArray a = z.getColumn(i).sub(means.getFloat(i));
            variances.putScalar(i, a.mul(a).sumNumber().floatValue() / rows);
        }

        return variances;
    }

    public INDArray normalize(INDArray z) {
        int rows = z.rows();
        int cols = z.columns();
        INDArray means = means(z);
        INDArray variances = variances(z);
        INDArray preSclShft = Nd4j.create(rows, cols);
        INDArray norm = Nd4j.create(rows, cols);

        for (int i = 0; i < cols; i++) {
            INDArray currCol = z.getColumn(i);
            INDArray normalizedCol = currCol.sub(means.getFloat(i)).div(Math.sqrt(variances.getFloat(i) + epsilon));
            preSclShft.putColumn(i, normalizedCol);
            norm.putColumn(i, normalizedCol.mul(scale.getFloat(i)).add(shift.getFloat(i)));
        }

        this.means = means;
        this.variances = variances;
        this.runningMeans = runningMeans.mul(momentum).add(means.mul((1 - momentum)));
        this.runningVariances = runningVariances.mul(momentum).add(variances.mul((1 - momentum)));
        this.preNormZ = z;
        this.preScaleShiftZ = preSclShft;

        return norm;
    }

    public void setPreNormZ(INDArray z) {
        this.preNormZ = z;
    }


    public INDArray gradientPreBN(INDArray dLdzHat) {
        int rows = dLdzHat.rows();
        int cols = dLdzHat.columns();
        INDArray dLdz = Nd4j.create(rows, cols);
        INDArray std = Transforms.pow(variances.add(epsilon), 0.5);
        INDArray scalingFactor = scale.div(std);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float gradientElement = dLdzHat.getFloat(i, j);
                float normElement = preScaleShiftZ.getFloat(i, j);
                dLdz.put(i, j,
                    scalingFactor.getFloat(i) * (gradientElement - 
                    ((gradientElement - normElement) / rows) * 
                    ((gradientElement * normElement) / rows)));
            }
            
        }

        return dLdz;
    }

    public INDArray gradientPreBNSimple(INDArray dLdzHat) {
        int rows = dLdzHat.rows();
        int cols = dLdzHat.columns();
        INDArray dLdz = Nd4j.create(rows, cols);
        INDArray std = Transforms.pow(variances.add(epsilon), 0.5);
        INDArray scalingFactor = scale.div(std);

        for (int i = 0; i < rows; i++) {
            dLdz.putRow(i, dLdzHat.getRow(i).reshape(dLdzHat.columns(), 1).mul(scalingFactor));
        }

        return dLdz;
    }
}
