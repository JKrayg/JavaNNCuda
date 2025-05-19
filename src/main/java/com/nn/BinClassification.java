package com.nn;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu.softplus;

import com.nn.activation.ReLU;
import com.nn.activation.Sigmoid;
import com.nn.activation.Softmax;
import com.nn.components.Layer;
import com.nn.components.NeuralNet;
import com.nn.layers.Dense;
import com.nn.layers.Output;
import com.nn.training.loss.BinCrossEntropy;
import com.nn.training.loss.CatCrossEntropy;
import com.nn.training.metrics.BinaryMetrics;
import com.nn.training.optimizers.Adam;
import com.nn.training.regularizers.L2;

public class BinClassification {
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
        Dense dense1 = new Dense(4, new ReLU(), 4);
        dense1.addRegularizer(new L2());
        Dense dense2 = new Dense(4, new ReLU());
        dense2.addRegularizer(new L2());
        Output out = new Output(data.getClasses().size(), new Softmax(), new CatCrossEntropy());

        nn.addLayer(dense1);
        nn.addLayer(dense2);
        nn.addLayer(out);

        nn.compile(new Adam(0.001), new BinaryMetrics());

        nn.miniBatchFit(data, 8, 10);

        // for (Layer l : nn.getLayers()) {
        //     if (l.getPreActivation() != null) {
        //         System.out.println("preactivation: " + l.getPreActivation().data());
        //     }
            
        //     System.out.println("activation: " + l.getActivations().data());
        //     System.out.println("weights: " + ((Dense) l).getWeights().data());
        //     System.out.println("bias: " + l.getBias().data());
        //     if (l instanceof Output) {
        //         System.out.println("pred: " + l.getActivations());
        //         System.out.println(Arrays.toString(((Output) l).getLabels().shape()));
        //         System.out.println("labels: " + ((Output) l).getLabels());
        //     }
        // }
    }

}
