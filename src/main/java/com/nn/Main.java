// import org.nd4j.linalg.api.ndarray.INDArray;
// import org.nd4j.linalg.factory.Nd4j;
// import org.nd4j.linalg.factory.Nd4jBackend;

// public class Main {
//     static {
//         System.setProperty("org.nd4j.linalg.defaultbackend", "org.nd4j.linalg.jcublas.JCublasBackend");
//     }

//     public static void main(String[] args) {
//         try {
//             Nd4jBackend backend = Nd4j.getBackend();
//             System.out.println("Loaded Backend: " + backend.getClass().getName());
//             System.out.println("Environment: " + Nd4j.getExecutioner().getEnvironmentInformation());
//             System.out.println("Devices: " + Nd4j.getExecutioner().getEnvironmentInformation().get("cuda.devicesInformation"));

//             float[] values = new float[]{1, 2, 3, 4};
//             INDArray array = Nd4j.create(values, new int[]{2, 2}, 'c');
//             INDArray result = array.add(1.0);
//             System.out.println("Array:\n" + array);
//             System.out.println("Array + 1:\n" + result);
//         } catch (Exception e) {
//             e.printStackTrace();
//         }
//     }

// }

// // Jake Krayger
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;
import java.io.File;
import java.io.FileNotFoundException;

import com.nn.Data;
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
        // String filePath = "src\\resources\\datasets\\wdbc.data";
        // String filePath = "src\\resources\\datasets\\iris.data";
        String filePath = "src\\resources\\datasets\\mnist.csv";
        ArrayList<double[]> dataArrayList = new ArrayList<>();
        // ArrayList<String> labelsArrayList = new ArrayList<>();
        ArrayList<Integer> labelsArrayList = new ArrayList<>();
        

        try {
            File f = new File(filePath);
            Scanner scan = new Scanner(f);
            while (scan.hasNextLine()) {
                // ** iris data **
                // String line = scan.nextLine();
                // String values = line.substring(0, line.lastIndexOf(","));
                // double[] toDub;
                // String[] splitValues = values.split(",");
                // toDub = new double[splitValues.length];

                // for (int i = 0; i < splitValues.length; i++) {
                //     toDub[i] = Double.parseDouble(splitValues[i]);
                // }

                // dataArrayList.add(toDub);
                // String label = line.substring(line.lastIndexOf(",") + 1);
                // labelsArrayList.add(label);


                // ** wdbc data **
                // String line = scan.nextLine();
                // String[] splitLine = line.split(",", 3);
                // String label = splitLine[1];
                // labelsArrayList.add(label);
                // double[] toDub;
                // String values = splitLine[2];
                // String[] splitValues = values.split(",");
                // toDub = new double[splitValues.length];

                // for (int i = 0; i < splitValues.length; i++) {
                //     toDub[i] = Double.parseDouble(splitValues[i]);
                // }

                // dataArrayList.add(toDub);

                // ** mnist **
                String line = scan.nextLine();
                String[] splitLine = line.split(",", 2);
                int label = Integer.parseInt(splitLine[0]);
                labelsArrayList.add(label);
                double[] toDub;
                String values = splitLine[1];
                String[] splitValues = values.split(",");
                toDub = new double[splitValues.length];

                for (int i = 0; i < splitValues.length; i++) {
                    toDub[i] = Double.parseDouble(splitValues[i]);
                }

                dataArrayList.add(toDub);

            }
            scan.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }


        double[][] data_ = dataArrayList.toArray(new double[0][]);
        // String[] labels = labelsArrayList.toArray(new String[0]);
        Integer[] labels = labelsArrayList.toArray(new Integer[0]);

        // System.out.println(new SimpleMatrix(data_).getRow(0));
        // System.out.println(labelsI[0]);

        // double[][] testerData = new double[75][];
        // String[] testerLabels = new String[75];
        // Random rand = new Random();

        // for (int i = 0; i < 75; i++) {
        //     int r = rand.nextInt(0, labels.length);
        //     testerData[i] = data_[r].clone();
        //     testerLabels[i] = labels[r];
        // }

        // Data data = new Data(testerData, testerLabels);

        Data data = new Data(data_, labels);
        data.minMaxNormalization();
        // data.zScoreNormalization();

        data.split(0.20, 0.20);

        NeuralNet nn = new NeuralNet();
        Dense d1 = new Dense(
            256,
            new ReLU(),
            784);
        d1.addRegularizer(new Dropout(0.2));
        // d1.addRegularizer(new L2(0.01));
        // d1.addNormalization(new BatchNormalization());

        Dense d2 = new Dense(
            128,
            new ReLU());
        d2.addRegularizer(new Dropout(0.2));
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
        nn.miniBatchFit(data.getTrainData(), data.getTestData(), data.getValData(), 32, 10);
        // SimpleMatrix dataa = data.getTrainData().extractMatrix(
        //         0, data.getTrainData().getNumRows(), 0, data.getTrainData().getNumCols() - (data.getClasses().size() > 2 ? data.getClasses().size() : 1));
        // SimpleMatrix labelss = data.getTrainData().extractMatrix(
        //         0, data.getTrainData().getNumRows(), data.getTrainData().getNumCols() - (data.getClasses().size() > 2 ? data.getClasses().size() : 1), data.getTrainData().getNumCols());
        // nn.forwardPass(dataa, labelss);
        
        // System.out.println("Initial output: " + d5.getActivations().extractVector(true, 0));
        // for (Layer l : nn.getLayers()) {
        //     System.out.println(l.getActivations());
            
        // }

        // BatchNormalization b = new BatchNormalization();
        // b.setScale(new SimpleMatrix(new double[]{1, 1, 1}));
        // b.setShift(new SimpleMatrix(new double[]{12, 12, 12}));
        // b.setMeans(new SimpleMatrix(new double[]{2, 2, 2}));
        // b.setVariances(new SimpleMatrix(new double[]{3, 3, 3}));
        // b.setPreNormZ(new SimpleMatrix(new double[][]{{13, 14, 15}, {16, 17, 18}, {19, 20, 21}, {22, 23, 24}}));
        // SimpleMatrix dLdzHat = new SimpleMatrix(new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}});
        // Dropout d = new Dropout(0.2);
        // System.out.println(dLdzHat);
        // System.out.println(d.regularize(dLdzHat));
        // // System.out.println(b.means(dLdzHat));
        // // b.setMeans(b.means(dLdzHat));
        // // System.out.println(b.variances(dLdzHat));
        // // System.out.println(b.getScale());
        // // System.out.println(b.getMeans());
        // // System.out.println(b.getVariances());
        // // System.out.println(b.getPreNormZ());
        // // System.out.println(dLdzHat);

        // System.out.println(b.gradientPreBN(dLdzHat));

    }
}
