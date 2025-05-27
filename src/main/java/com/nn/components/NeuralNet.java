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
import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu.shape_of;
import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu.test_scalar;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import com.nn.Data;
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
        if (layers == null) {
            layers = new ArrayList<>();
            // l.setPrev(null);
        } else {
            Layer prev = layers.get(layers.size() - 1);
            prev.setNext(l);
            l.setPrev(prev);
        }
        this.layers.add(l);
    }

    public void compile(Optimizer o, Metrics m) {
        this.optimizer = o;
        this.metrics = m;
    }

    public void miniBatchFit(Data data, int batchSize, int epochs) {
        INDArray trainData = data.getTrainData();
        INDArray trainLabels = data.getTrainLabels();
        INDArray testData = data.getTestData();
        INDArray testLabels = data.getTestLabels();
        INDArray valData = data.getValData();
        INDArray valLabels = data.getValLabels();

        for (int i = 0; i < layers.size(); i++) {
            Layer prev = null;
            Layer curr = layers.get(i);
            if (i != 0) {
                prev = layers.get(i - 1);
            }

            // init layers
            curr.initLayer(prev, batchSize);

            if (optimizer instanceof Adam) {
                curr.initForAdam();
            }

        }

        // callbacks [find a better way to do this]
        // if (callbacks != null) {
        //     for (Callback c : callbacks) {
        //         if (c instanceof EarlyStopping) {
        //             EarlyStopping es = (EarlyStopping) c;
        //             lossHistory = Nd4j.createUninitialized();
        //         }
        //         break;
        //     }
        // }

        boolean reshape = false;
        long[] shape = trainData.shape();

        // shuffle
        List<INDArray> arraysToShuffle;
        if (shape.length == 2) {
            if (trainLabels.shape().length == 1) {
                trainLabels = trainLabels.reshape(trainData.shape()[0], 1);
                valLabels = valLabels.reshape(valLabels.shape()[0], 1);
                testLabels = testLabels.reshape(testLabels.shape()[0], 1);
            }
            arraysToShuffle = Arrays.asList(trainData, trainLabels);
        } else {
            arraysToShuffle = Arrays.asList(trainData.reshape(shape[0], shape[2] * shape[3]), trainLabels);
            reshape = true;
        }

        // forward/backprop batches per epoch
        for (int i = 0; i < epochs; i++) {
            this.lossHistory = null;
            Nd4j.shuffle(arraysToShuffle, new Random(), 1);

            if (reshape) {
                trainData = trainData.reshape(shape);
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

        System.out.println("train metrics: ");
        metrics(trainData, trainLabels);
        System.out.println("val metrics: ");
        metrics(testData, testLabels);
        System.out.println("test metrics: ");
        metrics(valData, valLabels);

    }

    public void batchFit(INDArray train, INDArray test, INDArray validation, int epochs) {
        int cols = train.columns();
        int rows = train.rows();

        for (int i = 0; i < epochs; i++) {
            INDArray dater = train.get(NDArrayIndex.all(), NDArrayIndex.interval(0, cols
                    - numClasses));
            INDArray labels = train.get(NDArrayIndex.all(), NDArrayIndex.interval(cols -
                    numClasses, cols));
            forwardPass(dater, labels);
            // backprop(dater, labels);

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
        // metrics(testData, testLabels);
    }

    public void forwardPass(INDArray data, INDArray labels) {
        Layer dummy = new Layer();
        dummy.setActivations(data);
        layers.get(0).forwardProp(dummy);
        for (int q = 1; q < layers.size(); q++) {
            Layer curr = layers.get(q);
            Layer prev = layers.get(q - 1);
            // Layer prev = null;
            // if (q > 0) {
            //     prev = layers.get(q - 1);
            // }

            curr.forwardProp(prev);

            if (curr instanceof Output) {
                ((Output) curr).setLabels(labels);
            }
        }
    }

    public void backprop(INDArray data, INDArray labels) {
        Output outLayer = (Output) layers.get(layers.size() - 1);
        Loss lossFunc = outLayer.getLoss();
        INDArray loss = Nd4j.create(new float[] {
            lossFunc.execute(outLayer.getActivations(), outLayer.getLabels())
        });

        if (this.lossHistory == null) {
            this.lossHistory = loss;
        } else {
            this.lossHistory = Nd4j.hstack(this.lossHistory, loss);
        }

        INDArray gradientWrtOutput = lossFunc.gradient(outLayer, outLayer.getLabels());
        // recursively get gradients
        // getGradients(outLayer, gradientWrtOutput, data);
        Layer prev = layers.get(layers.indexOf(outLayer) - 1);
        outLayer.getGradients(prev, gradientWrtOutput, data);

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

    public void metrics(INDArray d, INDArray l) {
        forwardPass(d, l);
        Output outLayer = (Output) layers.get(layers.size() - 1);
        metrics.getMetrics(outLayer.getActivations(), l);
    }

    public float loss(INDArray d, INDArray l) {
        forwardPass(d, l);
        Output outLayer = (Output) layers.get(layers.size() - 1);
        // System.out.println(Arrays.toString(outLayer.getActivations().shape()));
        // System.out.println(Arrays.toString(outLayer.getLabels().shape()));
        // System.out.println(outLayer.getActivations().data());
        // System.out.println(outLayer.getLabels().data());

        return outLayer.getLoss().execute(outLayer.getActivations(), l);
    }

}