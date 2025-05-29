package com.nn.examples;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;
import com.nn.Data;
import com.nn.activation.ReLU;
import com.nn.activation.Softmax;
import com.nn.components.NeuralNet;
import com.nn.layers.Dense;
import com.nn.layers.Output;
import com.nn.training.callbacks.Callback;
import com.nn.training.callbacks.StepDecay;
import com.nn.training.loss.CatCrossEntropy;
import com.nn.training.metrics.BinaryMetrics;
import com.nn.training.metrics.MultiClassMetrics;
import com.nn.training.optimizers.Adam;

public class MultiClassification {
    // ** iris data ** --------------------------------------------------
    public static void main(String[] args) {
        String filePath = "src\\resources\\datasets\\iris.data";
        ArrayList<String> labelsArrayList = new ArrayList<>();
        ArrayList<float[]> dataArrayList = new ArrayList<>();

        try {
            File f = new File(filePath);
            Scanner scan = new Scanner(f);
            while (scan.hasNextLine()) {

                String line = scan.nextLine();
                String values = line.substring(0, line.lastIndexOf(","));
                float[] toDub;
                String[] splitValues = values.split(",");
                toDub = new float[splitValues.length];

                for (int i = 0; i < splitValues.length; i++) {
                    toDub[i] = Float.parseFloat(splitValues[i]);
                }

                dataArrayList.add(toDub);
                String label = line.substring(line.lastIndexOf(",") + 1);
                labelsArrayList.add(label);

            }
            scan.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        float[][] data_ = dataArrayList.toArray(new float[0][]);
        String[] labels = labelsArrayList.toArray(new String[0]);


        Data data = new Data(data_, labels);
        data.zScoreNormalization();
        data.shuffle();
        data.split(0.20, 0.20);

        NeuralNet nn = new NeuralNet();
        Dense dense1 = new Dense(16, new ReLU(), 4);
        Dense dense2 = new Dense(8, new ReLU());
        Output out = new Output(data.getClasses().size(), new Softmax(), new CatCrossEntropy());

        nn.addLayer(dense1);
        nn.addLayer(dense2);
        nn.addLayer(out);

        // nn.compile(new Adam(0.01), new BinaryMetrics());
        nn.optimizer(new Adam(0.01));
        nn.metrics(new MultiClassMetrics());
        nn.callbacks(new Callback[]{new StepDecay(0.05, 10)});

        nn.miniBatchFit(data, 16, 50);
    }

}
