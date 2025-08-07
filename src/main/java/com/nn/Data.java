package com.nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import com.nn.utils.MathUtils;

public class Data {
    private INDArray data;
    private INDArray target;
    private INDArray trainData;
    private INDArray trainTarget;
    private INDArray testData;
    private INDArray testTarget;
    private INDArray valData;
    private INDArray valTarget;
    private HashMap<String, Integer> classes;
    // private HashMap<String, Float> classes2;

    public Data() {
    }

    public Data(float[][] data) {
        this.data = Nd4j.create(data);
    }

    public Data(float[][] data, String[] target) {
        this.data = Nd4j.create(data);

        // create a hashtable of target mapped to an integer
        HashMap<String, Integer> h = new HashMap<>();
        Set<String> c = new HashSet<>(List.of(target));
        int count = 0;
        for (String s : c) {
            h.put(s, count);
            count++;
        }

        this.classes = h;

        // create list of target values
        float[] ls = new float[target.length];
        for (int i = 0; i < target.length; i++) {
            ls[i] = classes.get(target[i]);
        }

        if (classes.size() > 2) {
            this.target = oneHot(Nd4j.create(ls));
        } else {
            this.target = Nd4j.create(ls);
        }

    }

    public Data(float[][] data, float[] targetVals) {
        this.data = Nd4j.create(data);
        this.target = Nd4j.create(targetVals);
    }

    public Data(float[][] data, Integer[] target) {
        // System.out.println(target.length);
        this.data = Nd4j.create(data);

        // create a hashtable of target mapped to an integer
        HashMap<String, Integer> h = new HashMap<>();
        int idx = 0;
        for (int i : target) {
            h.put(Integer.toString(i), idx);
            idx++;
        }

        this.classes = h;

        // create list of target values
        float[] ls = new float[target.length];
        for (int i = 0; i < target.length; i++) {
            ls[i] = classes.get(Integer.toString(target[i]));
        }

        // System.out.println(ls.length);

        if (classes.size() > 2) {
            this.target = oneHot(Nd4j.create(ls));
        } else {
            this.target = Nd4j.create(ls);
        }

    }

    public Data(INDArray data, INDArray target) {
        // long[] dataShape = data.shape();
        // if (dataShape[1] == 1) {
        // data = data.reshape(dataShape[0], dataShape[2], dataShape[3]);
        // }

        this.data = data;

        // create a hashtable of target mapped to an integer
        HashMap<String, Integer> h = new HashMap<>();
        for (int i : target.toIntVector()) {
            h.put(Integer.toString(i), i);
        }

        this.classes = h;

        // create list of a target values
        // float[] ls = new float[target.length];
        // for (int i = 0; i < target.length; i++) {
        // ls[i] = classes.get(Integer.toString(target[i]));
        // }

        // System.out.println(ls.length);

        if (classes.size() > 2) {
            this.target = oneHot(target);
        } else {
            this.target = target;
        }

    }

    public void flatten() {
        this.data = data.reshape(
            data.size(0),
            data.size(1) * data.size(2));
    }

    public INDArray oneHot(INDArray target) {
        INDArray encoded = Nd4j.create(target.length(), classes.size());
        for (int i = 0; i < target.length(); i++) {
            INDArray curr = Nd4j.create(1, classes.size());
            curr.putScalar((int) target.getFloat(i), 1.0);
            encoded.putRow(i, curr);
        }

        return encoded;
    }

    public INDArray getData() {
        return data;
    }

    public INDArray getTarget() {
        return target;
    }

    public INDArray getTestData() {
        return testData;
    }

    public INDArray getTestTarget() {
        return testTarget;
    }

    public INDArray getTrainData() {
        return trainData;
    }

    public INDArray getTrainTarget() {
        return trainTarget;
    }

    public INDArray getValData() {
        return valData;
    }

    public INDArray getValTarget() {
        return valTarget;
    }

    public HashMap<String, Integer> getClasses() {
        return classes;
    }

    public void zScoreNormalization() {
        MathUtils maths = new MathUtils();
        int cols = data.columns();
        int rows = data.rows();
        if (data != null) {
            for (int i = 0; i < cols; i++) {
                INDArray col = data.getColumn(i);
                float mean = (col.sumNumber().floatValue() / rows);
                float std = maths.std(col);
                for (int j = 0; j < rows; j++) {
                    data.putScalar(j, i, (data.getFloat(j, i) - mean) / std);
                }
            }

        }
    }

    public void minMaxNormalization() {
        if (data != null) {
            float max = data.maxNumber().floatValue();
            float min = data.minNumber().floatValue();
            data.subi(min).divi(max - min);
        }
    }

    // only works for <= 3D
    public void shuffle() {
        List<INDArray> arraysToShuffle;
        boolean reshape = false;
        long[] shape = data.shape();

        if (data.shape().length == target.shape().length) {
            arraysToShuffle = Arrays.asList(data, target);
        } else {
            arraysToShuffle = Arrays.asList(data.reshape(shape[0], shape[1] * shape[2]), target);
            reshape = true;
        }

        Nd4j.shuffle(arraysToShuffle, 1);
        if (reshape) {
            data = data.reshape(shape[0], shape[1], shape[2]);
        }
    }

    public void split(double testSize, double valSize) {
        int rows = (int) data.size(0);
        int testSetSize = (int) (rows * testSize);
        int valSetSize = (int) (rows * valSize);
        int trainSetSize = rows - (testSetSize + valSetSize);

        this.trainData = data
                .get(NDArrayIndex.interval(0, trainSetSize));
        this.testData = data
                .get(NDArrayIndex.interval(trainSetSize, trainSetSize + testSetSize));
        this.valData = data
                .get(NDArrayIndex.interval(trainSetSize + testSetSize, rows));
        this.trainTarget = target
                .get(NDArrayIndex.interval(0, trainSetSize));
        this.testTarget = target
                .get(NDArrayIndex.interval(trainSetSize, trainSetSize + testSetSize));
        this.valTarget = target
                .get(NDArrayIndex.interval(trainSetSize + testSetSize, rows));

    }
}
