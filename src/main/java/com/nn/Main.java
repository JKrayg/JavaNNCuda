package com.nn;


// Jake Krayger
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu.conv2d;
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
    public static float[][][][] loadMnist(String path, int maxImages) throws IOException {
        DataInputStream dis = new DataInputStream(new FileInputStream(path));
        int magic = dis.readInt();
        if (magic != 0x00000803) throw new IOException("Invalid MNIST image file");

        int numImages = dis.readInt();
        int numRows = dis.readInt();
        int numCols = dis.readInt();

        int imageCount = Math.min(numImages, maxImages);
        float[][][][] images = new float[imageCount][1][numRows][numCols];

        byte[] buffer = new byte[numRows * numCols];
        for (int i = 0; i < imageCount; i++) {
            dis.readFully(buffer);
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < numCols; c++) {
                    images[i][0][r][c] = buffer[r * numCols + c] & 0xFF;
                }
            }
}
        dis.close();
        return images;
    }

    public static float[] loadMnistLabels(String path, int maxLabels) throws IOException {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(path)))) {
            int magic = dis.readInt();
            if (magic != 0x00000801) throw new IOException("Invalid MNIST label file");

            int numLabels = dis.readInt();
            int labelCount = Math.min(numLabels, maxLabels);

            byte[] buffer = new byte[labelCount];
            dis.readFully(buffer);

            float[] labels = new float[labelCount];
            for (int i = 0; i < labelCount; i++) {
                labels[i] = buffer[i] & 0xFF;
            }

            dis.close();
            return labels;
        }
    }

    public static INDArray loadCifar10Images(String filePath, int maxImages) throws IOException {
        int imageSize = 32 * 32 * 3;
        int recordSize = 1 + imageSize; // 1 byte for label
        FileInputStream fis = new FileInputStream(filePath);
        DataInputStream dis = new DataInputStream(fis);
    
        float[][][][] images = new float[maxImages][32][32][3]; // [numImages][pixels][channel]
        
        for (int i = 0; i < maxImages; i++) {
            dis.readUnsignedByte(); // Skip the label
    
            byte[] imageBytes = new byte[imageSize];
            dis.readFully(imageBytes);
    
            // Separate channels
            for (int channel = 0; channel < 3; channel++) {
                for (int j = 0; j < 1024; j++) {
                    int idx = channel * 1024 + j;
                    int row = j / 32;
                    int col = j % 32;
                    images[i][row][col][channel] = (imageBytes[idx] & 0xFF);
                }
            }
        }
    
        dis.close();
        return Nd4j.create(images);
    }

    public static INDArray loadCifar10Labels(String filePath, int maxImages) throws IOException {
        int recordSize = 1 + 3072;
        FileInputStream fis = new FileInputStream(filePath);
        DataInputStream dis = new DataInputStream(fis);
    
        float[] labels = new float[maxImages];
        
        for (int i = 0; i < maxImages; i++) {
            labels[i] = dis.readUnsignedByte();
            dis.skipBytes(3072); // Skip image data
        }
    
        dis.close();
        return Nd4j.create(labels);
    }

    // public static void showImageFromINDArray(INDArray image, int scale) {
    //     // Validate shape: Expecting [batchSize, height, width, channels]
    //     if (image.rank() != 4) {
    //         throw new IllegalArgumentException("Expected 4D array [batchSize, height, width, channels], got rank: " + image.rank());
    //     }
    //     int batchSize = (int) image.shape()[0];
    //     if (batchSize != 1) {
    //         throw new IllegalArgumentException("Expected batch size of 1, got: " + batchSize);
    //     }
    //     int height = (int) image.shape()[1];   // 32
    //     int width = (int) image.shape()[2];    // 32
    //     int channels = (int) image.shape()[3]; // 3 (RGB)
    
    //     if (channels != 3) {
    //         throw new IllegalArgumentException("Expected 3 channels (RGB), got: " + channels);
    //     }
    
    //     // Create BufferedImage
    //     BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
    
    //     for (int y = 0; y < height; y++) {
    //         for (int x = 0; x < width; x++) {
    //             // Extract RGB from the single image (batch index 0)
    //             int r = (int) (image.getDouble(0, y, x, 0) * 255); // Red channel
    //             int g = (int) (image.getDouble(0, y, x, 1) * 255); // Green channel
    //             int b = (int) (image.getDouble(0, y, x, 2) * 255); 
    
    //             // Ensure RGB values are within [0, 255] range
    //             r = Math.min(255, Math.max(0, r));
    //             g = Math.min(255, Math.max(0, g));
    //             b = Math.min(255, Math.max(0, b));
    
    //             int rgb = (r << 16) | (g << 8) | b; // Combine RGB channels
    //             img.setRGB(x, y, rgb);
    //         }
    //     }
    
    //     // Scale the image for better visibility
    //     Image scaledImage = img.getScaledInstance(width * scale, height * scale, Image.SCALE_FAST);
    
    //     // Display in JFrame
    //     JFrame frame = new JFrame();
    //     frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    //     frame.getContentPane().add(new JLabel(new ImageIcon(scaledImage)));
    //     frame.pack();
    //     frame.setVisible(true);
    // }

    
    public static void showImageFromINDArray(INDArray image, int scale) {
        // Determine if the input is 3D [height, width, channels] or 4D [batchSize, height, width, channels]
        int rank = image.rank();
        if (rank != 3 && rank != 4) {
            throw new IllegalArgumentException("Expected 3D array [height, width, channels] or 4D array [batchSize, height, width, channels], got rank: " + rank);
        }

        // Extract dimensions
        int batchSize, height, width, channels;
        boolean is4D = (rank == 4);
        if (is4D) {
            batchSize = (int) image.shape()[0];
            if (batchSize != 1) {
                throw new IllegalArgumentException("Expected batch size of 1, got: " + batchSize);
            }
            height = (int) image.shape()[1];   // 28 for MNIST
            width = (int) image.shape()[2];    // 28 for MNIST
            channels = (int) image.shape()[3]; // 1 for grayscale
        } else {
            height = (int) image.shape()[0];   // 28 for MNIST
            width = (int) image.shape()[1];    // 28 for MNIST
            channels = (int) image.shape()[2]; // 1 for grayscale
        }

        // Validate channels for grayscale
        if (channels != 1) {
            throw new IllegalArgumentException("Expected 1 channel (grayscale), got: " + channels);
        }

        // Create BufferedImage
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Extract grayscale value
                double value = is4D ? image.getDouble(0, y, x, 0) : image.getDouble(y, x, 0);

                // Ensure value is within [0, 255] range
                int gray = (int) Math.min(255, Math.max(0, value));

                // For grayscale, R = G = B
                int rgb = (gray << 16) | (gray << 8) | gray;
                img.setRGB(x, y, rgb);
            }
        }

        // Scale the image for better visibility
        Image scaledImage = img.getScaledInstance(width * scale, height * scale, Image.SCALE_FAST);

        // Display in JFrame
        JFrame frame = new JFrame();
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.getContentPane().add(new JLabel(new ImageIcon(scaledImage)));
        frame.pack();
        frame.setVisible(true);
    }

    // public static INDArray padImages(INDArray images) {
    //     INDArray padded = Nd4j.zeros(images.size(0),
    //                                 images.size(1) + 1 * 2,
    //                                 images.size(2) + 1 * 2,
    //                                 images.size(3));
    //     padded.get(
    //         NDArrayIndex.all(),
    //         NDArrayIndex.interval(1, 29),
    //         NDArrayIndex.interval(1, 29),
    //         NDArrayIndex.all()
    //     ).assign(images);

    //     return padded;
    
    // }

    public static void main(String[] args) throws IOException {
        // mnist -----------------------------------------------------------
        // private static String mnistFolder = "src\\resources\\datasets\\mnist\\";
        // private static String trainImages = mnistFolder + "train-images.idx3-ubyte";
        // private static String trainLabels = mnistFolder + "train-labels.idx3-ubyte";
        // private static String testImages = mnistFolder + "t10k-images.idx3-ubyte";
        // private static String testLabels = mnistFolder + "t10k-labels.idx3-ubyte";
        float[][][][] testImages = loadMnist("src\\resources\\datasets\\mnist\\train-images.idx3-ubyte", 60000);
        float[] testLabels = loadMnistLabels("src\\resources\\datasets\\mnist\\train-labels.idx1-ubyte", 60000);
        INDArray data_ = Nd4j.create(testImages);
        INDArray labels = Nd4j.create(testLabels);
        // ------------------------------------------------------------------

        // cifar10 -----------------------------------------------------------
        // String cifarFolder = "src\\resources\\datasets\\cifar-10\\";
        // String[] files = {"data_batch_1.bin",
        //                   "data_batch_2.bin",
        //                   "data_batch_3.bin",
        //                   "data_batch_4.bin",
        //                   "data_batch_5.bin"};
        // INDArray data_ = Nd4j.create(50000, 32, 32, 3);
        // INDArray labels = Nd4j.create(50000);
        // int batch = 1;
        // for (String s: files) {
        //     INDArray imgs = loadCifar10Images(cifarFolder + s, 10000);
        //     INDArray labs = loadCifar10Labels(cifarFolder + s, 10000);
        //     data_.put(
        //         new INDArrayIndex[] {NDArrayIndex.interval(batch * 10000 - 10000, batch * 10000),
        //                              NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()},
        //                              imgs);

        //     labels.put(new INDArrayIndex[] {NDArrayIndex.interval(batch * 10000 - 10000, batch * 10000)}, labs);
        //     batch += 1;
        // }

        // -------------------------------------------------------------------

        // ** iris data ** --------------------------------------------------
        // String filePath = "src\\resources\\datasets\\iris.data";
        // ArrayList<String> labelsArrayList = new ArrayList<>();
        // ArrayList<float[]> dataArrayList = new ArrayList<>();

        // try {
        //     File f = new File(filePath);
        //     Scanner scan = new Scanner(f);
        //     while (scan.hasNextLine()) {
        //         
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


        // float[][] data_ = dataArrayList.toArray(new float[0][]);
        // String[] labels = labelsArrayList.toArray(new String[0]);

        // float[][] testerData = new float[45][];
        // Integer[] testerLabels = new Integer[45];
        // Random rand = new Random();

        // for (int i = 0; i < 45; i++) {
        //     int r = rand.nextInt(0, labels.length);
        //     testerData[i] = data_[r].clone();
        //     testerLabels[i] = labels[r];
        // }

        // Data data = new Data(testerData, testerLabels);
        //----------------------------------------------------------------------------


        Data data = new Data(data_, labels);
        // data.flatten();
        data.minMaxNormalization();
        // // data.zScoreNormalization();

        // only works for <= 3D right now
        // data.shuffle(data_, labels);

        data.split(0.20, 0.20);

        // INDArray batchImg = data.getTrainData().get(NDArrayIndex.interval(0, 1));
        // INDArray batchLbl = data.getTrainLabels().get(NDArrayIndex.interval(0, 1));
        // System.out.println(Arrays.toString(batchImg.shape()));
        // // System.out.println(batchImg.data());
        // System.out.println(batchLbl.argMax());

        // // INDArray images = data;
        // if (1 != 0) {
        //     batchImg = padImages(batchImg);
        //     batchImg = batchImg.reshape(batchImg.size(1), batchImg.size(1));
        // }

        // System.out.println(Arrays.toString(batchImg.shape()));
        // System.out.println(batchImg);
        // System.out.println(batchLbl.argMax());
        // System.out.println(Arrays.toString(batchImg.shape()));
        // System.out.println(singleLbl);
        // showImageFromINDArray(singleImg, 4);

        // // long totalStart = System.nanoTime();
        NeuralNet nn = new NeuralNet();
        Conv2d d1 = new Conv2d(
            10,
            new int[]{1, 28, 28},
            new int[]{3, 3},
            1,
            "same",
            new ReLU());

        Conv2d d2 = new Conv2d(
            20,
            new int[]{3, 3},
            1,
            "same",
            new ReLU());

        // Conv2d d3 = new Conv2d(
        //     40,
        //     new int[]{3, 3},
        //     1,
        //     "same",
        //     new ReLU());

        Flatten d3 = new Flatten();
        // Dense d1 = new Dense(256, new ReLU(), 784);
        Dense d4 = new Dense(128, new ReLU());

        Output d5 = new Output(
            data.getClasses().size(),
            new Softmax(),
            new CatCrossEntropy());

        nn.addLayer(d1);
        nn.addLayer(d2);
        nn.addLayer(d3);
        nn.addLayer(d4);
        nn.addLayer(d5);
        // nn.addLayer(d6);

        nn.compile(new Adam(0.001), new MultiClassMetrics());

        long totalStart = System.nanoTime();

        nn.miniBatchFit(data, 32, 1);

        double totalTimeMs = (System.nanoTime() - totalStart) / 1e6;
        System.out.println("Total mini batch Time: " + totalTimeMs + " ms");

        // INDArray patch = Nd4j.zeros(784, 90);
        // INDArray filter = Nd4j.zeros(90, 20);
        // System.out.println(Arrays.toString(patch.mmul(filter).shape()));


        // long totalStart = System.nanoTime();
        // double totalTimeMs = (System.nanoTime() - totalStart) / 1e6;
        // System.out.println("Total main Time: " + totalTimeMs + " ms");

    }
}
