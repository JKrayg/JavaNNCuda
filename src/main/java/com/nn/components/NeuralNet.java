package com.nn.components;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
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
    private LinkedList<Float> lossHistory;
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
        if (callbacks != null) {
            for (Callback c : callbacks) {
                if (c instanceof EarlyStopping) {
                    EarlyStopping es = (EarlyStopping) c;
                    lossHistory = new LinkedList<Float>();
                }
                break;
            }
        }
    }

    public void miniBatchFit(INDArray train, INDArray test, INDArray validation, int batchSize, int epochs) {
        // shuffle data and get new batches of size batchSize for each epoch
        System.out.println("minibatch");
        for (int i = 0; i < epochs; i++) {
            System.out.println("epoch: " + i);
            ArrayList<INDArray> shuffled = new ArrayList<>();
            ArrayList<INDArray> batchesData = new ArrayList<>();
            ArrayList<INDArray> batchesLabels = new ArrayList<>();
            System.out.println("here1");
            for (int j = 0; j < train.rows(); j++) {
                shuffled.add(train.getRow(j));
            }
            System.out.println("here2");
            Collections.shuffle(shuffled);

            System.out.println("here3");
            for (int k = 0; k < shuffled.size() / batchSize; k++) {
                INDArray currBatch = Nd4j.create(batchSize, train.columns());
                // System.out.println(currBatch.dataType()); //--------------------------------------
                int count = 0;
                for (int p = k * batchSize; p < k * batchSize + batchSize; p++) {
                    currBatch.putRow(count, shuffled.get(p));
                    count += 1;
                }
                batchesData.add(currBatch.get(
                        NDArrayIndex.interval(0, currBatch.rows()), NDArrayIndex.interval(0, currBatch.columns() - (numClasses > 2 ? numClasses : 1))));
                batchesLabels.add(currBatch.get(
                    NDArrayIndex.interval(0, currBatch.rows()), NDArrayIndex.interval(currBatch.columns() - (numClasses > 2 ? numClasses : 1),
                        currBatch.columns())));
            }
            System.out.println("here4");

            // last batch - find a better way
            if (shuffled.size() % batchSize > 0) {
                INDArray lastBatch = Nd4j.create(shuffled.size() % batchSize, train.columns());
                int count = 0;
                System.out.println("here5");
                for (int m = batchSize * (shuffled.size() / batchSize); m < shuffled.size(); m++) {
                    lastBatch.putRow(count, shuffled.get(m));
                    count += 1;
                }
                System.out.println("here6");
                batchesData.add(lastBatch.get(
                    NDArrayIndex.interval(0, lastBatch.rows()), NDArrayIndex.interval(0, lastBatch.columns() - (numClasses > 2 ? numClasses : 1))));
                batchesLabels.add(lastBatch.get(
                    NDArrayIndex.interval(0, lastBatch.rows()), NDArrayIndex.interval(lastBatch.columns() - (numClasses > 2 ? numClasses : 1),
                        lastBatch.columns())));
            }

            // do below for each batch
            System.out.println("here7");
            for (int v = 0; v < batchesData.size(); v++) {
                forwardPass(batchesData.get(v), batchesLabels.get(v));
                backprop(batchesData.get(v), batchesLabels.get(v));
                System.out.println("here" + (8+v));

                if (optimizer instanceof Adam) {
                    ((Adam) optimizer).updateCount();
                }
            }

            System.out.println("herelast");

            this.loss = loss(train);
            this.valLoss = loss(validation);
            System.out.println("loss: " + loss + " - val loss: " + valLoss);
        }
        // INDArray testData = test.get(0, test.rows(), 0,
        //         test.columns() - (numClasses > 2 ? numClasses : 1));
        // INDArray testLabels = test.get(0, test.rows(),
        //         test.columns() - (numClasses > 2 ? numClasses : 1), test.columns());

        // forwardPass(testData, testLabels);
        // Output outLayer = (Output) layers.get(layers.size() - 1);
        // // System.out.println("Prediction : one hot label");
        // // for (int h = 0; h < testData.rows(); h++) {
        // //     System.out.print(outLayer.getActivations().getRow(h).concatColumns(testLabels.getRow(h)));
        // // }

        // System.out.println("Prediction : True Value");
        //  for (int h = 0; h < testData.rows(); h++) {
        //      System.out.print(outLayer.getActivations().get(h));
        //      System.out.print(" : " + testLabels.get(h));
        //      System.out.println();
        //  }

        // test
        System.out.println();
        System.out.println("train metrics: ");
        metrics(train);
        System.out.println("val metrics: ");
        metrics(validation);
        System.out.println("test metrics: ");
        metrics(test);

    }

    public void batchFit(INDArray train, INDArray test, INDArray validation, int epochs) {
        for (int i = 0; i < epochs; i++) {
            INDArray data = train.get(
                NDArrayIndex.interval(0, train.rows()), NDArrayIndex.interval(0,
                    train.columns() - (numClasses > 2 ? numClasses : 1)));
            INDArray labels = train.get(
                NDArrayIndex.interval(0, train.rows()), NDArrayIndex.interval(train.columns() - (numClasses > 2 ? numClasses : 1), train.columns()));

            forwardPass(data, labels);
            backprop(data, labels);

            if (optimizer instanceof Adam) {
                ((Adam) optimizer).updateCount();
            }

            this.loss = loss(train);
            this.valLoss = loss(validation);
            System.out.println("loss: " + loss + " - val loss: " + valLoss);
        }

        System.out.println("train metrics: ");
        metrics(train);
        System.out.println("test metrics: ");
        metrics(test);
    }

    public void forwardPass(INDArray data, INDArray labels) {
        Layer L1 = layers.get(0);
        INDArray zL1 = maths.weightedSum(data, L1);
        ActivationFunction actF = L1.getActFunc();
        L1.setPreActivations(zL1);

        // normalize before activation if batch normalization
        Normalization nL1 = L1.getNormalization();
        if (nL1 instanceof BatchNormalization) {
            zL1 = nL1.normalize(zL1);
        }
        
        INDArray aL1 = actF.execute(zL1);

        // dropout [find a better way to do this]
        if (L1.getRegularizers() != null) {
            for (Regularizer r : L1.getRegularizers()) {
                if (r instanceof Dropout) {
                    aL1 = r.regularize(aL1);
                }
                break;
            }
        }
        
        L1.setActivations(aL1);
        

        for (int q = 1; q < layers.size(); q++) {
            Layer curr = layers.get(q);
            Layer prev = layers.get(q - 1);
            INDArray z = maths.weightedSum(prev, curr);
            Normalization norm = curr.getNormalization();
            ActivationFunction actFunc = curr.getActFunc();
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
        INDArray gradientWrtOutput = outLayer.getLoss().gradient(outLayer, outLayer.getLabels());

        // recursively get gradients
        getGradients(outLayer, gradientWrtOutput, data);

        // update weights/biases
        for (Layer l : layers) {
            l.updateWeights(optimizer);
            l.updateBiases(optimizer);

            // update beta/gamma if batch normalzation
            Normalization norm = l.getNormalization();
            if (norm instanceof BatchNormalization) {
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
        Normalization norm = curr.getNormalization();
        INDArray grad;
        if (norm instanceof BatchNormalization) {
            BatchNormalization batchNorm = (BatchNormalization) norm;
            grad = batchNorm.gradientPreBN(gradient);
            batchNorm.setGradientShift(batchNorm.gradientShift(gradient));
            batchNorm.setGradientScale(batchNorm.gradientScale(gradient));
        } else {
            grad = gradient.dup();
        }

        // weights/bias gradients
        if (currLayer instanceof Output) {
            Output out = (Output) curr;
            Layer prev = layers.get(layers.indexOf(curr) - 1);
            gradientWrtWeights = out.gradientWeights(prev, gradient);
            gradientWrtBias = out.gradientBias(gradient);
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

    public void metrics(INDArray test) {
        INDArray testData = test.get(
            NDArrayIndex.interval(0, test.rows()), NDArrayIndex.interval(0, test.columns() - (numClasses > 2 ? numClasses : 1)));
        INDArray testLabels = test.get(
            NDArrayIndex.interval(0, test.rows()), NDArrayIndex.interval(test.columns() - (numClasses > 2 ? numClasses : 1), test.columns()));

        forwardPass(testData, testLabels);
        Output outLayer = (Output) layers.get(layers.size() - 1);
        metrics.getMetrics(outLayer.getActivations(), testLabels);
    }

    public float loss(INDArray d) {
        INDArray data = d.get(
            NDArrayIndex.interval(0, d.rows()), NDArrayIndex.interval(0, d.columns() - (numClasses > 2 ? numClasses : 1)));
        INDArray labels = d.get(
            NDArrayIndex.interval(0, d.rows()), NDArrayIndex.interval(d.columns() - (numClasses > 2 ? numClasses : 1), d.columns()));

        forwardPass(data, labels);
        Output outLayer = (Output) layers.get(layers.size() - 1);
        
        return outLayer.getLoss().execute(outLayer.getActivations(), labels);
    }

    

}