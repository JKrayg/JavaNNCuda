package com.nn;


// Jake Krayger
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import com.nn.activation.*;
import com.nn.components.*;
import com.nn.layers.*;
import com.nn.training.loss.*;
import com.nn.training.metrics.*;
import com.nn.training.normalization.BatchNormalization;
import com.nn.training.optimizers.*;
import com.nn.training.regularizers.*;

public class Main {
    public static void main(String[] args) {
        // long totalStart = System.nanoTime();
        // String filePath = "src\\resources\\datasets\\wdbc.data";
        // String filePath = "src\\resources\\datasets\\iris.data";
        String filePath = "src\\resources\\datasets\\mnist.csv";
        // String mnistFolder = "src\\resources\\datasets\\mnist\\";
        // String trainImages = mnistFolder + "train-images.idx3-ubyte";
        // String trainLabels = mnistFolder + "train-labels.idx3-ubyte";
        // String testImages = mnistFolder + "t10k-images.idx3-ubyte";
        // String testLabels = mnistFolder + "t10k-labels.idx3-ubyte";
        ArrayList<float[]> dataArrayList = new ArrayList<>();
        // ArrayList<String> labelsArrayList = new ArrayList<>();
        ArrayList<Integer> labelsArrayList = new ArrayList<>();
        

        try {
            File f = new File(filePath);
            Scanner scan = new Scanner(f);
            while (scan.hasNextLine()) {
                // ** iris data ** -----------------------------------------------
                // String line = scan.nextLine();
                // String values = line.substring(0, line.lastIndexOf(","));
                // float[] toDub;
                // String[] splitValues = values.split(",");
                // toDub = new float[splitValues.length];

                // for (int i = 0; i < splitValues.length; i++) {
                //     toDub[i] = float.parsefloat(splitValues[i]);
                // }

                // dataArrayList.add(toDub);
                // String label = line.substring(line.lastIndexOf(",") + 1);
                // labelsArrayList.add(label);


                // ** wdbc data ** ------------------------------------------------
                // String line = scan.nextLine();
                // String[] splitLine = line.split(",", 3);
                // String label = splitLine[1];
                // labelsArrayList.add(label);
                // float[] toDub;
                // String values = splitLine[2];
                // String[] splitValues = values.split(",");
                // toDub = new float[splitValues.length];

                // for (int i = 0; i < splitValues.length; i++) {
                //     toDub[i] = Float.parseFloat(splitValues[i]);
                // }

                // dataArrayList.add(toDub);

                // ** mnist ** -----------------------------------------------------
                String line = scan.nextLine();
                String[] splitLine = line.split(",", 2);
                int label = Integer.parseInt(splitLine[0]);
                labelsArrayList.add(label);
                float[] toDub;
                String values = splitLine[1];
                String[] splitValues = values.split(",");
                toDub = new float[splitValues.length];

                for (int i = 0; i < splitValues.length; i++) {
                    toDub[i] = Float.parseFloat(splitValues[i]);
                }

                dataArrayList.add(toDub);

            }
            scan.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }


        float[][] data_ = dataArrayList.toArray(new float[0][]);
        // String[] labels = labelsArrayList.toArray(new String[0]);
        Integer[] labels = labelsArrayList.toArray(new Integer[0]);

        // float[][] testerData = new float[45][];
        // Integer[] testerLabels = new Integer[45];
        // Random rand = new Random();

        // for (int i = 0; i < 45; i++) {
        //     int r = rand.nextInt(0, labels.length);
        //     testerData[i] = data_[r].clone();
        //     testerLabels[i] = labels[r];
        // }

        // Data data = new Data(testerData, testerLabels);

        Data data = new Data(data_, labels);
        data.minMaxNormalization();
        // data.zScoreNormalization();

        

        data.split(0.15, 0.15);

        
        // double totalTimeMs = (System.nanoTime() - totalStart) / 1e6;
        // System.out.println("Total data prep Time: " + totalTimeMs + " ms");

        // long totalStart = System.nanoTime();
        NeuralNet nn = new NeuralNet();
        Dense d1 = new Dense(
            256,
            new ReLU(),
            784);
        // d1.addRegularizer(new Dropout(0.2));
        // d1.addRegularizer(new L2(0.01));
        // d1.addNormalization(new BatchNormalization());

        Dense d2 = new Dense(
            128,
            new ReLU());
        // d2.addRegularizer(new Dropout(0.2));
        // d2.addRegularizer(new L2(0.01));
        // d2.addNormalization(new BatchNormalization());

        Output d3 = new Output(
            data.getClasses().size(),
            new Softmax(),
            new CatCrossEntropy());
        // d3.addRegularizer(new L2(0.01));

        nn.addLayer(d1);
        nn.addLayer(d2);
        nn.addLayer(d3);
        nn.compile(new Adam(0.001), new MultiClassMetrics());
        
        long totalStart = System.nanoTime();

        nn.miniBatchFit(data.getTrainData(), data.getTestData(), data.getValData(), 32, 5);

        double totalTimeMs = (System.nanoTime() - totalStart) / 1e6;
        System.out.println("Total mini batch Time: " + totalTimeMs + " ms");


        // long totalStart = System.nanoTime();
        // double totalTimeMs = (System.nanoTime() - totalStart) / 1e6;
        // System.out.println("Total main Time: " + totalTimeMs + " ms");

    }
}
