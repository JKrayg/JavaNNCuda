package com.nn.examples;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import com.nn.Data;
import com.nn.activation.ReLU;
import com.nn.activation.Softmax;
import com.nn.components.NeuralNet;
import com.nn.layers.Dense;
import com.nn.layers.Output;
import com.nn.training.callbacks.Callback;
import com.nn.training.callbacks.EarlyStopping;
import com.nn.training.callbacks.StepDecay;
import com.nn.training.loss.CatCrossEntropy;
import com.nn.training.metrics.BinaryMetrics;
import com.nn.training.metrics.MultiClassMetrics;
import com.nn.training.normalization.BatchNormalization;
import com.nn.training.optimizers.Adam;
import com.nn.training.regularizers.Dropout;
import com.nn.training.regularizers.L2;

public class MultiClassification {
    // ** iris data ** --------------------------------------------------
    public static void main(String[] args) {
        String filePathR = "src\\resources\\datasets\\winequality-red.csv";
        String filePathW = "src\\resources\\datasets\\winequality-white.csv";
        // String filePath = "src\\resources\\datasets\\iris.data";
        ArrayList<String> labelsArrayList = new ArrayList<>();
        // ArrayList<Integer> labelsArrayList = new ArrayList<>();
        ArrayList<float[]> dataArrayList = new ArrayList<>();

        // iris
        // try {
        //     File f = new File(filePath);
        //     Scanner scan = new Scanner(f);

        //     while (scan.hasNextLine()) {

        //         String line = scan.nextLine();
        //         String values = line.substring(0, line.lastIndexOf(","));
        //         float[] toDub;
        //         String[] splitValues = values.split(",");
        //         toDub = new float[splitValues.length];

        //         for (int i = 0; i < splitValues.length; i++) {
        //             toDub[i] = Float.parseFloat(splitValues[i]);
        //         }

        //         dataArrayList.add(toDub);
        //         String label = line.substring(line.lastIndexOf(",") + 1);
        //         labelsArrayList.add(label);

        //     }
        //     scan.close();
        // } catch (FileNotFoundException e) {
        //     e.printStackTrace();
        // }


        // wine
        try {
            File fr = new File(filePathR);
            File fw = new File(filePathW);
            Scanner scan = new Scanner(fr);
            scan.nextLine();

            while (scan.hasNextLine()) {

                String line = scan.nextLine();
                String values = line.substring(0, line.lastIndexOf(";"));
                float[] toDub;
                String[] splitValues = values.split(";");
                toDub = new float[splitValues.length];

                for (int i = 0; i < splitValues.length; i++) {
                    toDub[i] = Float.parseFloat(splitValues[i]);
                }

                dataArrayList.add(toDub);
                String label = line.substring(line.lastIndexOf(";") + 1);
                labelsArrayList.add(label);

            }
            scan.close();

            Scanner scan2 = new Scanner(fw);

            scan2.nextLine();
            while (scan2.hasNextLine()) {

                String line = scan2.nextLine();
                String values = line.substring(0, line.lastIndexOf(";"));
                float[] toDub;
                String[] splitValues = values.split(";");
                toDub = new float[splitValues.length];

                for (int i = 0; i < splitValues.length; i++) {
                    toDub[i] = Float.parseFloat(splitValues[i]);
                }

                dataArrayList.add(toDub);
                String label = line.substring(line.lastIndexOf(";") + 1);
                labelsArrayList.add(label);

            }
            scan2.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        float[][] data_ = dataArrayList.toArray(new float[0][]);
        // Integer[] labels = labelsArrayList.toArray(new Integer[0]);
        String[] labels = labelsArrayList.toArray(new String[0]);

        System.out.println(data_.length);
        System.out.println(labels.length);


        Data data = new Data(data_, labels);
        data.zScoreNormalization();
        data.shuffle();
        data.split(0.20, 0.20);
        

        NeuralNet nn = new NeuralNet();
        Dense dense1 = new Dense(64, new ReLU(), 11);
        dense1.addNormalization(new BatchNormalization());
        Dense dense2 = new Dense(32, new ReLU());
        Dense dense3 = new Dense(16, new ReLU());
        Output out = new Output(data.getClasses().size(), new Softmax(), new CatCrossEntropy());

        nn.addLayer(dense1);
        nn.addLayer(dense2);
        nn.addLayer(dense3);
        nn.addLayer(out);

        nn.optimizer(new Adam(0.001));
        nn.metrics(new MultiClassMetrics());
        nn.callbacks(new Callback[]{new EarlyStopping("val_loss", 0.001, 10)});

        nn.miniBatchFit(data, 32, 2);
    }

}
