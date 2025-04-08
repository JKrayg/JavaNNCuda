package com.nn.training.optimizers;

import java.sql.BatchUpdateException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import com.nn.components.Layer;
import com.nn.training.normalization.BatchNormalization;
import com.nn.training.normalization.Normalization;

public class Adam extends Optimizer {
    private float learningRate;
    private float momentumDecay = 0.9f;
    private float varianceDecay = 0.999f;
    private float epsilon = 1e-8f;
    private int updateCount = 1;

    public Adam(double learningRate) {
        this.learningRate = (float) learningRate;
    }
    
    public void setMomentumDecay(double md) {
        this.momentumDecay = (float) md;
    }

    public void setVarianceDecay(double vd) {
        this.varianceDecay = (float) vd;
    }

    public void setEpsilon(float epsilon) {
        this.epsilon = epsilon;
    }
    
    public INDArray executeWeightsUpdate(Layer l) {
        INDArray gWrtW = l.getGradientWeights();
        // System.out.println("weights gradient:");
        // System.out.println(gWrtW);
        INDArray momentumOfWeights = l.getWeightsMomentum()
                                         .mul(momentumDecay)
                                         .add(gWrtW.mul(1 - momentumDecay));

        INDArray varianceOfWeights = l.getWeightsVariance()
                                         .mul(varianceDecay)
                                         .add(Transforms.pow(gWrtW, 2).mul(1 - varianceDecay));

        INDArray currWeights = l.getWeights();
        INDArray biasCorrectedMomentum = momentumOfWeights.div(1 - Math.pow(momentumDecay, updateCount));
        INDArray biasCorrectedVariance = varianceOfWeights.div(1 - Math.pow(varianceDecay, updateCount));
        INDArray biasCorrection = biasCorrectedMomentum
                                      .div(Transforms.pow(biasCorrectedVariance, 0.5).add(epsilon))
                                      .mul(learningRate);

        INDArray updatedWeights = currWeights.sub(biasCorrection);

        l.setWeightsMomentum(momentumOfWeights);
        l.setWeightsVariance(varianceOfWeights);

        return updatedWeights;
    }

    public INDArray executeBiasUpdate(Layer l) {
        INDArray gWrtB = l.getGradientBias();
        // System.out.println(gWrtB.rows() + "x" + gWrtB.columns());
        int rows = gWrtB.rows();
        int cols = gWrtB.columns();
        // System.out.println("bias gradient:");
        // System.out.println(gWrtB);
        // System.out.println(l.getBiasMomentum().mul(momentumDecay).rows() + 
        //     " x " + l.getBiasMomentum().mul(momentumDecay).columns());
        // System.out.println(gWrtB.mul(1 - momentumDecay).rows() + 
        //     " x " + gWrtB.mul(1 - momentumDecay).columns());
        INDArray momentumOfBiases = l.getBiasMomentum()
                                        .mul(momentumDecay)
                                        .add(gWrtB.mul(1 - momentumDecay));

        INDArray varianceOfBias = l.getBiasVariance()
                                      .mul(varianceDecay)
                                      .add(Transforms.pow(gWrtB, 2).mul(1 - varianceDecay));

        INDArray currBiases = l.getBias();
        INDArray biasCorrectedMomentum = momentumOfBiases.div(1 - Math.pow(momentumDecay, updateCount));
        INDArray biasCorrectedVariance = varianceOfBias.div(1 - Math.pow(varianceDecay, updateCount));
        INDArray biasCorrection = biasCorrectedMomentum
                                      .div(Transforms.pow(biasCorrectedVariance,0.5).add(epsilon))
                                      .mul(learningRate);

        INDArray updatedBiases = currBiases.sub(biasCorrection);

        l.setBiasesMomentum(momentumOfBiases);
        l.setBiasesVariance(varianceOfBias);

        return updatedBiases;
    }

    public INDArray executeShiftUpdate(Normalization n) {
        INDArray gWrtSh = n.getGradientShift();
        // System.out.println("shift gradient:");
        // System.out.println(gWrtSh);
        INDArray momentumOfShifts = n.getShiftMomentum()
                                         .mul(momentumDecay)
                                         .add(gWrtSh.mul(1 - momentumDecay));

        INDArray varianceOfShifts = n.getShiftVariance()
                                         .mul(varianceDecay)
                                         .add(Transforms.pow(gWrtSh,2).mul(1 - varianceDecay));

        INDArray currShifts = n.getShift();
        INDArray biasCorrectedMomentum = momentumOfShifts.div(1 - Math.pow(momentumDecay, updateCount));
        INDArray biasCorrectedVariance = varianceOfShifts.div(1 - Math.pow(varianceDecay, updateCount));
        INDArray biasCorrection = biasCorrectedMomentum
                                      .div(Transforms.pow(biasCorrectedVariance,0.5).add(epsilon))
                                      .mul(learningRate);

        INDArray updatedShifts = currShifts.sub(biasCorrection);

        n.setShiftMomentum(momentumOfShifts);
        n.setShiftVariance(varianceOfShifts);

        return updatedShifts;
    }

    public INDArray executeScaleUpdate(Normalization n) {
        INDArray gWrtSc = n.getGradientScale();
        // System.out.println("scale gradient:");
        // System.out.println(gWrtSc);
        INDArray momentumOfScale = n.getScaleMomentum()
                                         .mul(momentumDecay)
                                         .add(gWrtSc.mul(1 - momentumDecay));

        INDArray varianceOfScale = n.getScaleVariance()
                                         .mul(varianceDecay)
                                         .add(Transforms.pow(gWrtSc,2).mul(1 - varianceDecay));

        INDArray currScale = n.getScale();
        INDArray biasCorrectedMomentum = momentumOfScale.div(1 - Math.pow(momentumDecay, updateCount));
        INDArray biasCorrectedVariance = varianceOfScale.div(1 - Math.pow(varianceDecay, updateCount));
        INDArray biasCorrection = biasCorrectedMomentum
                                      .div(Transforms.pow(biasCorrectedVariance,0.5).add(epsilon))
                                      .mul(learningRate);

        INDArray updatedScale = currScale.sub(biasCorrection);

        n.setScaleMomentum(momentumOfScale);
        n.setScaleVariance(varianceOfScale);

        return updatedScale;
    }

    public float getLearningRate() {
        return learningRate;
    }

    public float getMomentumDecay() {
        return momentumDecay;
    }

    public float getVarianceDecay() {
        return varianceDecay;
    }

    public float getEpsilon() {
        return epsilon;
    }

    public void updateCount() {
        this.updateCount += 1;
    }
}
