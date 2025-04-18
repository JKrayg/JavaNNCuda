package com.nn;


// Jake Krayger
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
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
    private static String mnistFolder = "src\\resources\\datasets\\mnist\\";
    private static String trainImages = mnistFolder + "train-images.idx3-ubyte";
    private static String trainLabels = mnistFolder + "train-labels.idx3-ubyte";
    private static String testImages = mnistFolder + "t10k-images.idx3-ubyte";
    private static String testLabels = mnistFolder + "t10k-labels.idx3-ubyte";
    public static float[][][] loadMnist(String path, int maxImages) throws IOException {
        DataInputStream dis = new DataInputStream(new FileInputStream(path));
        int magic = dis.readInt();
        if (magic != 0x00000803) throw new IOException("Invalid MNIST image file");

        int numImages = dis.readInt();
        int numRows = dis.readInt();
        int numCols = dis.readInt();

        int imageCount = Math.min(numImages, maxImages);
        float[][][] images = new float[imageCount][numRows][numCols];

        for (int i = 0; i < imageCount; i++) {
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < numCols; c++) {
                    images[i][r][c] = dis.readUnsignedByte(); // Normalize
                }
            }
        }

        dis.close();
        return images;
    }

    public static float[] loadMnistLabels(String path, int maxLabels) throws IOException {
        DataInputStream dis = new DataInputStream(new FileInputStream(path));
        int magic = dis.readInt();
        if (magic != 0x00000801) throw new IOException("Invalid MNIST label file");
    
        int numLabels = dis.readInt();
        int labelCount = Math.min(numLabels, maxLabels);
        float[] labels = new float[labelCount];
    
        for (int i = 0; i < labelCount; i++) {
            labels[i] = dis.readUnsignedByte(); // Labels are from 0–9
        }
    
        dis.close();
        return labels;
    }
    
    public static void main(String[] args) throws IOException {
        // float[][][] testImages = loadMnist("src\\resources\\datasets\\mnist\\train-images.idx3-ubyte", 60000);
        // float[] testLabels = loadMnistLabels("src\\resources\\datasets\\mnist\\train-labels.idx1-ubyte", 60000);
        // INDArray data_ = Nd4j.create(testImages);
        // INDArray labels = Nd4j.create(testLabels);
        
        String filePath = "src\\resources\\datasets\\iris.data";
        ArrayList<String> labelsArrayList = new ArrayList<>();
        ArrayList<float[]> dataArrayList = new ArrayList<>();

        try {
            File f = new File(filePath);
            Scanner scan = new Scanner(f);
            while (scan.hasNextLine()) {
                // ** iris data ** -----------------------------------------------
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
        // data.flatten();
        // data.minMaxNormalization();
        data.zScoreNormalization();

        

        data.split(0.10, 0.10);

        // System.out.println(Arrays.toString(data.getTrainData().shape()));
        // System.out.println(Arrays.toString(data.getTrainLabels().shape()));
        // System.out.println(Arrays.toString(data.getTestData().shape()));
        // System.out.println(Arrays.toString(data.getTestLabels().shape()));
        // System.out.println(Arrays.toString(data.getValData().shape()));
        // System.out.println(Arrays.toString(data.getValLabels().shape()));

        // System.out.println(data.getValData().get(NDArrayIndex.interval(0, 10)));
        // System.out.println(data.getValLabels().get(NDArrayIndex.interval(0, 10)));

        // long totalStart = System.nanoTime();
        NeuralNet nn = new NeuralNet();
        Dense d1 = new Dense(
            16,
            new ReLU(),
            4);
        // d1.addRegularizer(new Dropout(0.2));
        // d1.addRegularizer(new L2(0.01));
        // d1.addNormalization(new BatchNormalization());

        Dense d2 = new Dense(
            8,
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

        // nn.miniBatchFit(data.getTrainData(), data.getTrainLabels(),
        //                 data.getTestData(), data.getTestLabels(),
        //                 data.getValData(), data.getValLabels(), 16, 3);

        double totalTimeMs = (System.nanoTime() - totalStart) / 1e6;
        System.out.println("Total mini batch Time: " + totalTimeMs + " ms");


        // long totalStart = System.nanoTime();
        // double totalTimeMs = (System.nanoTime() - totalStart) / 1e6;
        // System.out.println("Total main Time: " + totalTimeMs + " ms");

    }
}
