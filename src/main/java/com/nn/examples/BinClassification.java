package com.nn.examples;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

import com.nn.Data;
import com.nn.activation.ReLU;
import com.nn.activation.Sigmoid;
import com.nn.components.NeuralNet;
import com.nn.layers.Dense;
import com.nn.layers.Output;
import com.nn.training.callbacks.Callback;
import com.nn.training.callbacks.StepDecay;
import com.nn.training.loss.BinCrossEntropy;
import com.nn.training.metrics.BinaryMetrics;
import com.nn.training.metrics.MultiClassMetrics;
import com.nn.training.optimizers.Adam;

public class BinClassification {
    public static void main(String[] args) {
        String filePath = "src\\resources\\datasets\\wdbc.data";
        ArrayList<String> labelsArrayList = new ArrayList<>();
        ArrayList<float[]> dataArrayList = new ArrayList<>();

        try {
            File f = new File(filePath);
            Scanner scan = new Scanner(f);
            while (scan.hasNextLine()) {

                // ** wdbc data **
                String line = scan.nextLine();
                String[] splitLine = line.split(",", 3);
                String label = splitLine[1];
                labelsArrayList.add(label);
                float[] toDub;
                String values = splitLine[2];
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
        String[] labels = labelsArrayList.toArray(new String[0]);

        Data data = new Data(data_, labels);
        data.zScoreNormalization();
        data.split(0.20, 0.20);

        NeuralNet nn = new NeuralNet();
        Dense dense1 = new Dense(16, new ReLU(), 30);
        Dense dense2 = new Dense(8, new ReLU());
        Output out = new Output(1, new Sigmoid(), new BinCrossEntropy());

        nn.addLayer(dense1);
        nn.addLayer(dense2);
        nn.addLayer(out);

        // nn.compile(new Adam(0.001), new BinaryMetrics());

        // CompileBuilder cb = nn.getCompileBuilder();
        nn.optimizer(new Adam(0.001));
        nn.metrics(new BinaryMetrics());
        // cb.callbacks(new Callback[]{new StepDecay(0.05, 10)});
        // cb.build();

        nn.miniBatchFit(data, 16, 20);
    }

}
