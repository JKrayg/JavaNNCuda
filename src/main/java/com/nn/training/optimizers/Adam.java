package com.nn.training.optimizers;

import java.sql.BatchUpdateException;
import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import com.nn.components.Layer;
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
        float momBiasCor = 1 - (float) Math.pow(momentumDecay, updateCount);
        float varBiasCor = 1 - (float) Math.pow(varianceDecay, updateCount);
        Layer lyr = l;

        // System.out.println("================");
        // System.out.println(lyr.getClass().getSimpleName());
        // System.out.println("mom: " + Arrays.toString(lyr.getWeightsMomentum().shape()));
        // System.out.println("weights grad: " + Arrays.toString(lyr.getGradientWeights().shape()));
        // System.out.println();

        
        // System.out.println(Arrays.toString(lyr.getWeightsVariance().mul(varianceDecay).shape()));
        // System.out.println(Arrays.toString(lyr.getGradientWeights().mul(1 - momentumDecay).shape()));
        // System.out.println();

        lyr.setWeightsMomentum(lyr.getWeightsMomentum().mul(momentumDecay)
                            .addi(lyr.getGradientWeights().mul(1 - momentumDecay)));
        lyr.setWeightsVariance(lyr.getWeightsVariance().mul(varianceDecay)
                            .addi(lyr.getGradientWeights().mul(lyr.getGradientWeights()).muli(1 - varianceDecay)));

        return lyr.getWeights().subi(lyr.getWeightsMomentum().div(momBiasCor)
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
