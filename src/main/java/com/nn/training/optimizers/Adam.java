package com.nn.training.optimizers;

import java.sql.BatchUpdateException;
import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import com.nn.components.Layer;
import com.nn.layers.Conv2d;
import com.nn.layers.Dense;
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

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
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
    
    public INDArray executeWeightsUpdate(Layer l) {
        float momBiasCor = 1 - (float) Math.pow(momentumDecay, updateCount);
        float varBiasCor = 1 - (float) Math.pow(varianceDecay, updateCount);
        Layer lyr = l;

        lyr.setWeightsMomentum(lyr.getWeightsMomentum().mul(momentumDecay)
                            .addi(lyr.getGradientWeights().mul(1 - momentumDecay)));
        lyr.setWeightsVariance(lyr.getWeightsVariance().mul(varianceDecay)
                            .addi(lyr.getGradientWeights().mul(lyr.getGradientWeights()).muli(1 - varianceDecay)));

        INDArray w = lyr.getWeights();
        if (lyr instanceof Conv2d) {
            w = lyr.getWeights().reshape(lyr.getWeights().shape()[0], -1);
        }

        // System.out.println(l.getClass().getSimpleName());

        return w.subi(lyr.getWeightsMomentum().div(momBiasCor)
                             .divi(Transforms.pow(lyr.getWeightsVariance().div(varBiasCor), 0.5).addi(epsilon))
                             .muli(learningRate));
    }


    public INDArray executeBiasUpdate(Layer l) {
        float momBiasCor = 1 - (float) Math.pow(momentumDecay, updateCount);
        float varBiasCor = 1 - (float) Math.pow(varianceDecay, updateCount);
        
        l.setBiasesMomentum(l.getBiasMomentum().mul(momentumDecay)
                            .addi(l.getGradientBias().mul(1 - momentumDecay)));
        l.setBiasesVariance(l.getBiasVariance().mul(varianceDecay)
                            .addi(l.getGradientBias().mul(l.getGradientBias()).muli(1 - varianceDecay)));


        return l.getBias().subi(l.getBiasMomentum().div(momBiasCor)
                          .divi(Transforms.pow(l.getBiasVariance().div(varBiasCor),0.5).addi(epsilon))
                          .muli(learningRate));
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
        // INDArray biasCorrectedMomentum = momentumOfShifts.div(1 - Math.pow(momentumDecay, updateCount));
        // INDArray biasCorrectedVariance = varianceOfShifts.div(1 - Math.pow(varianceDecay, updateCount));
        INDArray biasCorrection = momentumOfShifts.div(1 - Math.pow(momentumDecay, updateCount))
                .div(Transforms.pow(varianceOfShifts.div(1 - Math.pow(varianceDecay, updateCount)),0.5).add(epsilon))
                .mul(learningRate);

        INDArray updatedShifts = currShifts.sub(biasCorrection);

        n.setShiftMomentum(momentumOfShifts);
        n.setShiftVariance(varianceOfShifts);

        return updatedShifts;
    }

    public INDArray executeScaleUpdate(Normalization n) {
        INDArray gWrtSc = n.getGradientScale();
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

}
