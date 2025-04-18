package com.nn.components;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.custom.RandomShuffle;
import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu.flatten;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import com.nn.activation.*;
import com.nn.initialize.*;
import com.nn.layers.*;
import com.nn.training.loss.*;
import com.nn.training.metrics.Metrics;
import com.nn.training.normalization.*;
import com.nn.training.optimizers.*;
import com.nn.training.regularizers.*;
import com.nn.training.callbacks.*;
import com.nn.utils.MathUtils;

public class NeuralNet {
    private ArrayList<Layer> layers;
    private Optimizer optimizer;
    private Metrics metrics;
    private float loss;
    private float valLoss;
    private int numClasses;
    private MathUtils maths = new MathUtils();
    private INDArray lossHistory;
    private ArrayList<Callback> callbacks;

    public ArrayList<Layer> getLayers() {
        return layers;
    }

    public void addLayer(Layer l) {
        ActivationFunction actFunc = l.getActFunc();
        INDArray biases = Nd4j.create(l.getNumNeurons(), 1);
        if (this.layers != null) {
            Layer prevLayer = this.layers.get(this.layers.size() - 1);
            if (actFunc instanceof ReLU) {
                l.setWeights(new HeInit().initWeight(prevLayer, l));
                biases.addi(0.1);
                l.setBiases(biases);
            } else {
                l.setWeights(new GlorotInit().initWeight(prevLayer, l));
                l.setBiases(biases);
            }
        } else {
            layers = new ArrayList<>();
            if (actFunc instanceof ReLU) {
                l.setWeights(new HeInit().initWeight(l.getInputSize(), l));
                biases.addi(0.1);
                l.setBiases(biases);
            } else {
                l.setWeights(new GlorotInit().initWeight(l.getInputSize(), l));
                l.setBiases(biases);
            }
        }


        // init for batch normalization
        int numNeur = l.getNumNeurons();
        if (l.getNormalization() instanceof BatchNormalization) {
            BatchNormalization norm = (BatchNormalization) l.getNormalization();
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

        this.layers.add(l);

    }

    public void compile(Optimizer o, Metrics m) {
        this.optimizer = o;
        this.metrics = m;

        for (Layer lyr : layers) {
            if (lyr instanceof Output) {
                this.numClasses = lyr.getNumNeurons();
            }

            if (optimizer instanceof Adam) {
                INDArray weightsO = Nd4j.create(lyr.getWeights().rows(), lyr.getWeights().columns());
                INDArray biasO = Nd4j.create(lyr.getBias().rows(), lyr.getBias().columns());
                lyr.setWeightsMomentum(weightsO);
                lyr.setWeightsVariance(weightsO);
                lyr.setBiasesMomentum(biasO);
                lyr.setBiasesVariance(biasO);
                Normalization norm = lyr.getNormalization();
                if (norm != null) {
                    INDArray shiftO = Nd4j.create(norm.getShift().rows(), norm.getShift().columns());
                    INDArray scaleO = Nd4j.create(norm.getScale().rows(), norm.getScale().columns());
                    norm.setShiftMomentum(shiftO);
                    norm.setShiftVariance(shiftO);
                    norm.setScaleMomentum(scaleO);
                    norm.setScaleVariance(scaleO);
                }
            }
        }

        // callbacks [find a better way to do this]
        // if (callbacks != null) {
        //     for (Callback c : callbacks) {
        //         if (c instanceof EarlyStopping) {
        //             EarlyStopping es = (EarlyStopping) c;
        //             lossHistory = new LinkedList<Float>();
        //         }
        //         break;
        //     }
        // }
    }

    public void miniBatchFit(INDArray trainData, INDArray trainLabels,
                             INDArray testData, INDArray testLabels,
                             INDArray valData, INDArray valLabels,
                             int batchSize, int epochs) {

        boolean reshape = false;
        long[] shape = trainData.shape();

        List<INDArray> arraysToShuffle;
        if (trainData.shape().length == trainLabels.shape().length) {
            arraysToShuffle = Arrays.asList(trainData, trainLabels);
        } else {
            arraysToShuffle = Arrays.asList(trainData.reshape(shape[0], shape[1]*shape[2]), trainLabels);
            reshape = true;
        }

        for (int i = 0; i < epochs; i++) {
            this.lossHistory = null;

            Nd4j.shuffle(arraysToShuffle, new Random(), 1);

            if (reshape) {
                trainData = trainData.reshape(shape[0], shape[1], shape[2]);
            }

            // do below for each batch
            int rows = trainLabels.rows();
            INDArray dataBatch;
            INDArray labelsBatch;

            for (int k = 0; k < rows - (rows % batchSize); k += batchSize) {
                dataBatch = trainData.get(NDArrayIndex.interval(k, (k + batchSize)));
                labelsBatch = trainLabels.get(NDArrayIndex.interval(k, (k + batchSize)));
                forwardPass(dataBatch, labelsBatch);
                backprop(dataBatch, labelsBatch);

                if (optimizer instanceof Adam) {
                    ((Adam) optimizer).updateCount();
                }
            }

            // last batch
            if (rows % batchSize != 0) {
                dataBatch = trainData.get(NDArrayIndex.interval(batchSize - (rows % batchSize), batchSize));
                labelsBatch = trainLabels.get(NDArrayIndex.interval(batchSize - (rows % batchSize), batchSize));
                forwardPass(dataBatch, labelsBatch);
                backprop(dataBatch, labelsBatch);

                if (optimizer instanceof Adam) {
                    ((Adam) optimizer).updateCount();
                }
            }

            // print loss
            int numL = lossHistory.columns();
            this.loss = lossHistory.sumNumber().floatValue() / numL;
            this.valLoss = loss(valData, valLabels);

            System.out.println("loss: " + this.loss + " - val loss: " + this.valLoss);
        }

    }

    public void batchFit(INDArray train, INDArray test, INDArray validation, int epochs) {
        int cols = train.columns();
        int rows = train.rows();

        for (int i = 0; i < epochs; i++) {
            INDArray dater = train.get(NDArrayIndex.all(), NDArrayIndex.interval(0, cols - numClasses));
            INDArray labels = train.get(NDArrayIndex.all(), NDArrayIndex.interval(cols - numClasses, cols));
            forwardPass(dater, labels);
            backprop(dater, labels);

            if (optimizer instanceof Adam) {
                ((Adam) optimizer).updateCount();
            }

            // this.loss = loss(train);
            // this.valLoss = loss(validation);
            System.out.println("loss: " + loss + " - val loss: " + valLoss);
        }

        // System.out.println("train metrics: ");
        // metrics(train);
        // System.out.println("test metrics: ");
        // metrics(test);
    }

    public void forwardPass(INDArray data, INDArray labels) {
        // this implies dense layer. we need to change this to do
        // a proper forward pass depending on the type of layer and input data 
        for (int q = 0; q < layers.size(); q++) {
            Layer curr = layers.get(q);
            Layer prev;
            Normalization norm = curr.getNormalization();
            ActivationFunction actFunc = curr.getActFunc();
            INDArray z;

            if (q == 0) {
                z = maths.weightedSum(data, curr);
            } else {
                prev = layers.get(q - 1);
                z = maths.weightedSum(prev, curr);
            }
            
            curr.setPreActivations(z);
            
            // normalize before activation if batch normalization
            if (norm instanceof BatchNormalization) {
                z = norm.normalize(z);
            }

            INDArray activated = actFunc.execute(z); 

            // dropout [find a better way to do this]
            if (curr.getRegularizers() != null) {
                for (Regularizer r : curr.getRegularizers()) {
                    if (r instanceof Dropout) {
                        activated = r.regularize(activated);
                    }
                    break;
                }
            }
            
            curr.setActivations(activated);

            if (curr instanceof Output) {
                ((Output) curr).setLabels(labels);
            }
        }
    }

    public void backprop(INDArray data, INDArray labels) {
        Output outLayer = (Output) layers.get(layers.size() - 1);
        Loss lossFunc = outLayer.getLoss();
        INDArray loss = Nd4j.create(new float[]{lossFunc.execute(outLayer.getActivations(), outLayer.getLabels())});
        if (this.lossHistory == null) {
            this.lossHistory = loss;
        } else {
            this.lossHistory = Nd4j.hstack(this.lossHistory, loss);
        }

        INDArray gradientWrtOutput = lossFunc.gradient(outLayer, outLayer.getLabels());

        // recursively get gradients
        getGradients(outLayer, gradientWrtOutput, data);

        // update weights/biases
        for (Layer l : layers) {
            l.updateWeights(optimizer);
            l.updateBiases(optimizer);

            // update beta/gamma if batch normalzation
            if (l.getNormalization() instanceof BatchNormalization) {
                Normalization norm = l.getNormalization();
                ((BatchNormalization) norm).updateShift(optimizer);
                ((BatchNormalization) norm).updateScale(optimizer);
            }
        }
    }

    public void getGradients(Layer currLayer, INDArray gradient, INDArray data) {
        Layer curr = currLayer;
        INDArray gradientWrtWeights;
        INDArray gradientWrtBias;

        // batch normalization gradients
        Normalization norm = currLayer.getNormalization();
        INDArray grad;
        if (norm instanceof BatchNormalization) {
            BatchNormalization batchNorm = (BatchNormalization) norm;
            grad = batchNorm.gradientPreBNSimple(gradient);
            batchNorm.setGradientShift(batchNorm.gradientShift(gradient));
            batchNorm.setGradientScale(batchNorm.gradientScale(gradient));
        } else {
            grad = gradient;
        }

        // weights/bias gradients
        if (currLayer instanceof Output) {
            Layer prev = layers.get(layers.indexOf(currLayer) - 1);
            gradientWrtWeights = ((Output) currLayer).gradientWeights(prev, gradient);
            gradientWrtBias = ((Output) currLayer).gradientBias(gradient);
        } else {
            Layer prev;
            if (layers.indexOf(curr) > 0) {
                prev = layers.get(layers.indexOf(curr) - 1);
            } else {
                prev = new Layer();
                prev.setActivations(data);
            }

            gradientWrtWeights = currLayer.gradientWeights(prev, grad);
            gradientWrtBias = currLayer.gradientBias(grad);
            
        }

        // regularization (L1/L2) [find a better way to do this]
        if (curr.getRegularizers() != null) {
            for (Regularizer r : curr.getRegularizers()) {
                if (r instanceof L1 || r instanceof L2) {
                    gradientWrtWeights = gradientWrtWeights.add(r.regularize(currLayer.getWeights()));
                }
                break;
            }
        }


        curr.setGradientWeights(gradientWrtWeights);
        curr.setGradientBiases(gradientWrtBias);

        if (layers.indexOf(curr) > 0) {
            Layer prev = layers.get(layers.indexOf(curr) - 1);
            INDArray next = prev.getActFunc().gradient(prev, grad.mmul(currLayer.getWeights().transpose()));
            getGradients(prev, next, data);
        }
    }

    public void metrics(INDArray d, INDArray l) {
        forwardPass(d, l);
        Output outLayer = (Output) layers.get(layers.size() - 1);
        metrics.getMetrics(outLayer.getActivations(), l);
    }

    public float loss(INDArray d, INDArray l) {
        forwardPass(d, l);
        Output outLayer = (Output) layers.get(layers.size() - 1);
        
        return outLayer.getLoss().execute(outLayer.getActivations(), l);
    }

}