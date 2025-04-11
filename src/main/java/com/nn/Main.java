
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
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.exception.ND4JUnknownDataTypeException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

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
        ArrayList<float[]> dataArrayList = new ArrayList<>();
        // ArrayList<String> labelsArrayList = new ArrayList<>();
        ArrayList<Integer> labelsArrayList = new ArrayList<>();
        

        try {
            File f = new File(filePath);
            Scanner scan = new Scanner(f);
            while (scan.hasNextLine()) {
                // ** iris data **
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


                // ** wdbc data **
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

                // ** mnist **
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

        // int batchSize = 32; // Guess - adjust to yours
        // int features = 784; // E.g., MNIST
        // int hidden = 256;
        // INDArray inputs = Nd4j.rand(new int[]{1300 * batchSize, features});
        // INDArray weights1 = Nd4j.rand(new int[]{features, hidden});
        // INDArray weights2 = Nd4j.rand(new int[]{hidden, 10});

        // System.out.println("Batch Size: " + batchSize);
        // System.out.println("Input Shape: " + java.util.Arrays.toString(inputs.shape()));
        // System.out.println("Weights1 Shape: " + java.util.Arrays.toString(weights1.shape()));
        // System.out.println("Data Type: " + inputs.dataType());

        // long start = System.nanoTime();
        // for (int i = 0; i < 600; i++) {
        //     int startIdx = i * batchSize;
        //     INDArray batch = inputs.getRows(startIdx, startIdx + batchSize - 1);
        //     INDArray hiddenL = batch.mmul(weights1); // Forward
        //     INDArray output = hiddenL.mmul(weights2);
        //     INDArray grad = output.sub(1.0f); // Dummy backprop
        // }
        // long time = System.nanoTime() - start;
        // System.out.println("Time for 600 batches: " + time / 1e6 + " ms");

        // System.out.println(new INDArray(data_).getRow(0));
        // System.out.println(labelsI[0]);

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

        // float[][] allData = data.getData().toFloatMatrix(); // [600x784]
        // INDArray inputs = Nd4j.create(allData); // [600x784]
        // INDArray targets = data.getLabels();
        // // Weights
        // INDArray weights1 = Nd4j.rand(DataType.FLOAT, new int[]{784, 256});
        // INDArray weights2 = Nd4j.rand(DataType.FLOAT, new int[]{256, 10});

        // // Test epoch
        // long start = System.nanoTime();
        // INDArray hidden = inputs.mmul(weights1); // [600, 256]
        // INDArray output = hidden.mmul(weights2); // [600, 10]
        // INDArray gradOutput = output.sub(targets); // [600, 10]
        // INDArray gradHidden = gradOutput.mmul(weights2.transpose()); // [600, 256]
        // weights1.subi(inputs.transpose().mmul(gradHidden)); // [784, 256]
        // weights2.subi(hidden.transpose().mmul(gradOutput)); // [256, 10]
        // double timeMs = (System.nanoTime() - start) / 1e6;

        // // Output
        // System.out.println("Input Shape: " + Arrays.toString(inputs.shape()));
        // System.out.println("Target Shape: " + Arrays.toString(targets.shape()));
        // System.out.println("Data Type: " + inputs.dataType());
        // System.out.println("Epoch Time: " + timeMs + " ms");

        

        data.split(0.15, 0.15);

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
        INDArray testData = data.getTrainData().get(NDArrayIndex.interval(0, data.getTrainData().rows()),
            NDArrayIndex.interval(0, data.getTrainData().columns() - (data.getClasses().size() > 2 ? data.getClasses().size() : 1)));
        INDArray testLabels = data.getTrainData().get(NDArrayIndex.interval(0, data.getTrainData().rows()),
            NDArrayIndex.interval(data.getTrainData().columns() - (data.getClasses().size() > 2 ? data.getClasses().size() : 1), data.getTrainData().columns()));
        
        long totalStart = System.nanoTime();
        // long totalStartFor = System.nanoTime();
        // nn.forwardPass(testData, testLabels);
        // double totalTimeMsFor = (System.nanoTime() - totalStartFor) / 1e6;
        // System.out.println("Total forward Time: " + totalTimeMsFor + " ms");
        // long totalStartBack = System.nanoTime();
        // nn.backprop(testData, testLabels);
        // double totalTimeMsBack = (System.nanoTime() - totalStartBack) / 1e6;
        // System.out.println("Total back Time: " + totalTimeMsBack + " ms");
        
        nn.miniBatchFit(data.getTrainData(), data.getTestData(), data.getValData(), 32, 10);
        // nn.batchFit(data.getTrainData(), data.getTestData(), data.getValData(), 30);

        // INDArray s = Nd4j.create(new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11 , 12}, {13, 14, 15}});
        // INDArray s = Nd4j.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8});
        // System.out.println(s);
        // System.out.println(s.reshape(s.columns(), 1));
        // Nd4j.shuffle(s, new Random(), 1);
        // System.out.println(s);


        // long totalStart = System.nanoTime();
        double totalTimeMs = (System.nanoTime() - totalStart) / 1e6;
        System.out.println("Total main Time: " + totalTimeMs + " ms");

    }
}
