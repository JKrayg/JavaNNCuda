package com.nn.examples;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

import com.nn.Data;
import com.nn.activation.Linear;
import com.nn.activation.ReLU;
import com.nn.components.NeuralNet;
import com.nn.layers.Dense;
import com.nn.layers.Output;
import com.nn.training.loss.MSE;
import com.nn.training.metrics.MeanAbsoluteError;
import com.nn.training.normalization.BatchNormalization;
import com.nn.training.optimizers.Adam;
import com.nn.training.optimizers.SGD;
import com.nn.training.regularizers.Dropout;
import com.nn.training.regularizers.L2;

public class Regression {
    public static void main(String[] args) {
        // ** housing data ** --------------------------------------------------
        String filePath = "src\\resources\\datasets\\housing\\housing.data";
        ArrayList<String> lines = new ArrayList<>();

        try {

            File f = new File(filePath);
            Scanner scan = new Scanner(f);
            while (scan.hasNextLine()) {
                String line = scan.nextLine().trim();
                lines.add(line);
            }

            scan.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        float[][] data = new float[lines.size()][];
        float[] labels = new float[lines.size()];

        for (int j = 0; j < lines.size(); j++) {
            String currLine = lines.get(j);
            String[] splitValues = currLine.split("\\s+");
            float[] c = new float[splitValues.length];
            for (int i = 0 ; i < splitValues.length; i++) {
                c[i] = Float.parseFloat(splitValues[i]);
            }

            labels[j] = c[c.length - 1];
            float[] convFlt = new float[c.length - 1];
            for (int k = 0; k < c.length - 1; k++) {
                convFlt[k] = c[k];
            }

            data[lines.indexOf(currLine)] = convFlt;
        }

        Data data_ = new Data(data, labels);
        data_.zScoreNormalization();
        data_.split(0.2, 0.2);

        NeuralNet nn = new NeuralNet();
        Dense dense1 = new Dense(64, new ReLU(), 13);
        dense1.addRegularizer(new L2());
        dense1.addRegularizer(new Dropout(0.2));
        Dense dense2 = new Dense(32, new ReLU());
        dense2.addRegularizer(new L2());
        dense2.addRegularizer(new Dropout(0.2));
        Output out = new Output(1, new Linear(), new MSE());

        nn.addLayer(dense1);
        nn.addLayer(dense2);
        nn.addLayer(out);

        nn.compile(new Adam(0.001), new MeanAbsoluteError());

        nn.miniBatchFit(data_, 16, 100);

    }
}