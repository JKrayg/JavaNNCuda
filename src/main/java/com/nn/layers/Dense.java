package com.nn.layers;

import java.util.ArrayList;
import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.nn.activation.ActivationFunction;
import com.nn.activation.ReLU;
import com.nn.components.*;
import com.nn.initialize.GlorotInit;
import com.nn.initialize.HeInit;
import com.nn.training.loss.Loss;
import com.nn.training.normalization.BatchNormalization;
import com.nn.training.normalization.Normalization;
import com.nn.training.optimizers.Optimizer;
import com.nn.training.regularizers.Dropout;
import com.nn.training.regularizers.L1;
import com.nn.training.regularizers.L2;
import com.nn.training.regularizers.Regularizer;
import com.nn.utils.MathUtils;

// OutputLayer and Dense are pretty much the same class
public class Dense extends Layer {
    private int numNeurons;
    private INDArray weights;
    private INDArray weightsMomentum;
    private INDArray weightsVariance;
    private INDArray gradientWrtWeights;
    // private int numFeatures;
    // private int inputSize;

    public Dense() {}

    public Dense(int numNeurons) {
        this.numNeurons = numNeurons;
    }

    public Dense(int numNeurons, ActivationFunction actFunc) {
        super(actFunc);
        this.numNeurons = numNeurons;
    }

    public Dense(int numNeurons, ActivationFunction actFunc, int inputSize) {
        super(actFunc, inputSize);
        this.numNeurons = numNeurons;
        // this.numFeatures = inputSize;
    }

    // public int getNumFeatures() {
    //     return numFeatures;
    // }

    public INDArray getWeightsMomentum() {
        return weightsMomentum;
    }

    public INDArray getWeightsVariance() {
        return weightsVariance;
    }

    public INDArray getWeights() {
        return weights;
    }

    public int getNumNeurons() {
        return numNeurons;
    }

    public void setNumNeurons(int numNeurons) {
		this.numNeurons = numNeurons;
	}

    public void setGradientWeights(INDArray gWrtW) {
        this.gradientWrtWeights = gWrtW;
    }

    public void setWeightsVariance(INDArray v) {
        this.weightsVariance = v;
    }

    public void setWeightsMomentum(INDArray m) {
        this.weightsMomentum = m;
    }

    public void setWeights(INDArray weights) {
        this.weights = weights;
    }

    public void initForAdam() {
        INDArray weightsO = Nd4j.create(
                this.getWeights().rows(),
                this.getWeights().columns());
        INDArray biasO = Nd4j.create(
                this.getBias().rows(),
                this.getBias().columns());

        this.setWeightsMomentum(weightsO);
        this.setWeightsVariance(weightsO);
        this.setBiasesMomentum(biasO);
        this.setBiasesVariance(biasO);

        Normalization norm = this.getNormalization();
        if (norm != null) {
            INDArray shiftO = Nd4j.create(
                    norm.getShift().rows(),
                    norm.getShift().columns());
            INDArray scaleO = Nd4j.create(
                    norm.getScale().rows(),
                    norm.getScale().columns());

            norm.setShiftMomentum(shiftO);
            norm.setShiftVariance(shiftO);
            norm.setScaleMomentum(scaleO);
            norm.setScaleVariance(scaleO);
        }
    }

    public INDArray getGradientWeights() {
        return gradientWrtWeights;
    }

    public INDArray getGradient() {
        INDArray gradient = null;
        if (this instanceof Output) {
            gradient = ((Output) this).getLoss().gradient(this, ((Output) this).getLabels());
        } else {
            gradient = super.getActFunc().gradient(this, super.getPreActivation());
        }

        return gradient;
    }

    public INDArray gradientWeights(Layer prevLayer, INDArray gradient) {
        INDArray gWrtW = prevLayer.getActivations().transpose().mmul(gradient).div(prevLayer.getActivations().rows());
        return gWrtW;
    }

    public void updateWeights(Optimizer o) {
        this.weights = o.executeWeightsUpdate(this);
    }

    public Layer initLayer(Layer prev, int batchSize) {
        INDArray biases = Nd4j.create(numNeurons, 1);
        ActivationFunction actFunc = this.getActFunc();
        this.setActivations(Nd4j.create(batchSize, numNeurons));
        if (prev != null) {
            // INDArray prevAct = prev.getActivations();
            // this.setPreActivations(prev.getActivations());
            
            if (actFunc instanceof ReLU) {
                this.setWeights(new HeInit().initWeight(prev, this));
                biases.addi(0.1);
                this.setBiases(biases);
            } else {
                this.setWeights(new GlorotInit().initWeight(prev, this));
                this.setBiases(biases);
            }
            // this.setActivations(Nd4j.create(prevAct.shape()[0], numNeurons));
        } else {
            // this.setPreActivations(Nd4j.create(batchSize, numNeurons));
            if (actFunc instanceof ReLU) {
                this.setWeights(new HeInit().initWeight(this.getNumFeatures(), this));
                biases.addi(0.1);
                this.setBiases(biases);
            } else {
                this.setWeights(new GlorotInit().initWeight(this.getNumFeatures(), this));
                this.setBiases(biases);
            }
            
        }

        this.setPreActivations(Nd4j.create(batchSize, numNeurons));


        // init for batch normalization
        int numNeur = this.getNumNeurons();
        if (this.getNormalization() instanceof BatchNormalization) {
            BatchNormalization norm = (BatchNormalization) this.getNormalization();
            INDArray scVar = Nd4j.create(numNeur, 1);
            scVar.addi(1.0);
            INDArray shMeans = Nd4j.create(numNeur, 1);
            norm.setScale(scVar);
            norm.setShift(shMeans);
            norm.setMeans(shMeans);
            norm.setVariances(scVar);
            norm.setRunningMeans(shMeans);
            norm.setRunningVariances(scVar);
        }

        return this;

    }

    // public void forwardProp(Layer prev, INDArray data, INDArray labels) {
    //     // long totalStart1 = System.nanoTime();
    //     Normalization norm = this.getNormalization();
    //     ActivationFunction actFunc = this.getActFunc();
    //     MathUtils maths = new MathUtils();
    //     INDArray z;
    //     if (prev == null) {
    //         z = maths.weightedSum(data, this);
    //     } else {
    //         // System.out.println("=============" + Arrays.toString(prev.getActivations().shape()));
    //         z = maths.weightedSum(prev.getActivations(), this);
    //     }

    //     this.setPreActivations(z);

    //     // normalize before activation if batch normalization
    //     if (norm instanceof BatchNormalization) {
    //         z = norm.normalize(z);
    //     }

    //     INDArray activated = actFunc.execute(z);

    //     // dropout [find a better way to do this]
    //     if (this.getRegularizers() != null) {
    //         for (Regularizer r : this.getRegularizers()) {
    //             if (r instanceof Dropout) {
    //                 activated = r.regularize(activated);
    //             }
    //             break;
    //         }
    //     }

    //     this.setActivations(activated);

    //     // double totalTimeMs2 = (System.nanoTime() - totalStart1) / 1e6;
    //     // System.out.println("dense forward Time: " + totalTimeMs2 + " ms");
    // }

    // public void backprop(Layer prev, Layer curr, INDArray gradient, INDArray data) {
    //     // Output outLayer = (Output) layers.get(layers.size() - 1);
    //     // Loss lossFunc = outLayer.getLoss();
    //     // INDArray loss = Nd4j.create(new float[]{lossFunc.execute(outLayer.getActivations(), outLayer.getLabels())});
    //     // if (this.lossHistory == null) {
    //     //     this.lossHistory = loss;
    //     // } else {
    //     //     this.lossHistory = Nd4j.hstack(this.lossHistory, loss);
    //     // }

    //     // INDArray gradientWrtOutput = lossFunc.gradient(outLayer, outLayer.getLabels());

    //     // recursively get gradients
    //     this.getGradients(prev, this, gradient, data);

    //     // update weights/biases
    //     for (Layer l : layers) {
    //         ((Dense)l).updateWeights(optimizer);
    //         l.updateBiases(optimizer);

    //         // update beta/gamma if batch normalzation
    //         if (l.getNormalization() instanceof BatchNormalization) {
    //             Normalization norm = l.getNormalization();
    //             ((BatchNormalization) norm).updateShift(optimizer);
    //             ((BatchNormalization) norm).updateScale(optimizer);
    //         }
    //     }
    // }

    // public void getGradients(Layer prevLayer, Layer currLayer, INDArray gradient, INDArray data) {
    //     Layer curr = currLayer;
    //     INDArray gradientWrtWeights;
    //     INDArray gradientWrtBias;

    //     // batch normalization gradients
    //     Normalization norm = currLayer.getNormalization();
    //     INDArray grad;
    //     if (norm instanceof BatchNormalization) {
    //         BatchNormalization batchNorm = (BatchNormalization) norm;
    //         grad = batchNorm.gradientPreBNSimple(gradient);
    //         batchNorm.setGradientShift(batchNorm.gradientShift(gradient));
    //         batchNorm.setGradientScale(batchNorm.gradientScale(gradient));
    //     } else {
    //         grad = gradient;
    //     }

    //     // weights/bias gradients
    //     if (currLayer instanceof Output) {
    //         Layer prev = layers.get(layers.indexOf(currLayer) - 1);
    //         gradientWrtWeights = ((Output) currLayer).gradientWeights(prev, gradient);
    //         gradientWrtBias = ((Output) currLayer).gradientBias(gradient);
    //     } else {
    //         Layer prev;
    //         if (layers.indexOf(curr) > 0) {
    //             prev = layers.get(layers.indexOf(curr) - 1);
    //         } else {
    //             prev = new Layer();
    //             prev.setActivations(data);
    //         }

    //         gradientWrtWeights = ((Dense)currLayer).gradientWeights(prev, grad);
    //         gradientWrtBias = currLayer.gradientBias(grad);
            
    //     }

    //     // regularization (L1/L2) [find a better way to do this]
    //     if (curr.getRegularizers() != null) {
    //         for (Regularizer r : curr.getRegularizers()) {
    //             if (r instanceof L1 || r instanceof L2) {
    //                 gradientWrtWeights = gradientWrtWeights.add(r.regularize(((Dense)currLayer).getWeights()));
    //             }
    //             break;
    //         }
    //     }


    //     ((Dense)currLayer).setGradientWeights(gradientWrtWeights);
    //     curr.setGradientBiases(gradientWrtBias);

    //     if (layers.indexOf(curr) > 0) {
    //         Layer prev = layers.get(layers.indexOf(curr) - 1);
    //         INDArray next = prev.getActFunc().gradient(prev, grad.mmul(((Dense)currLayer).getWeights().transpose()));
    //         getGradients(prev, next, data);
    //     }
    // }
    
}