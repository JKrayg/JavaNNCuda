package com.nn;

import java.util.ArrayList;
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
    private INDArray labels;
    private INDArray train;
    private INDArray test;
    private INDArray validation;
    private HashMap<String, Integer> classes;

    public Data() {}

    public Data(float[][] data) {
        this.data = Nd4j.create(data);
    }

    public Data(float[][] data, String[] labels) {
        this.data = Nd4j.create(data);

        // create a hashtable of distinct labels mapped to an integer
        HashMap<String, Integer> h = new HashMap<>();
        Set<String> c = new HashSet<>(List.of(labels));
        int count = 0;
        for (String s: c) {
            h.put(s, count);
            count++;
        }

        this.classes = h;

        // create list of a label values
        float[] ls = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            ls[i] = classes.get(labels[i]);
        }

        if (classes.size() > 2) {
            this.labels = oneHot(Nd4j.create(ls));
        } else {
            this.labels = Nd4j.create(ls);
        }

        
    }

    public Data(float[][] data, Integer[] labels) {
        // System.out.println(labels.length);
        this.data = Nd4j.create(data);

        // create a hashtable of distinct labels mapped to an integer
        HashMap<String, Integer> h = new HashMap<>();
        for (int i: labels) {
            h.put(Integer.toString(i), i);
        }

        this.classes = h;

        // create list of a label values
        float[] ls = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            ls[i] = classes.get(Integer.toString(labels[i]));
        }

        // System.out.println(ls.length);

        if (classes.size() > 2) {
            this.labels = oneHot(Nd4j.create(ls));
        } else {
            this.labels = Nd4j.create(ls);
        }

        
    }

    public INDArray oneHot(INDArray labels) {
        INDArray encoded = Nd4j.create(labels.columns(), classes.size());
        for (int i = 0; i < labels.columns(); i++) {
            INDArray curr = Nd4j.create(1, classes.size());
            curr.putScalar((int)labels.getFloat(i), 1.0);
            encoded.putRow(i, curr);
        }

        return encoded;
    }

    public INDArray getData() {
        return data;
    }

    public INDArray getLabels() {
        return labels;
    }

    public INDArray getTestData() {
        return test;
    }

    public INDArray getTrainData() {
        return train;
    }

    public INDArray getValData() {
        return validation;
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
            data = data.div(max);
        }
    }

    public void split(double testSize, double valSize) {
        INDArray all = Nd4j.concat(1, data, labels);
        Nd4j.shuffle(all, new Random(), 1);
        int rows = all.rows();

        int testSetSize = (int) (rows * testSize);
        int valSetSize = (int) (rows * valSize);
        int trainSetSize = rows - (testSetSize + valSetSize);

        this.train = all.get(NDArrayIndex.interval(0,trainSetSize),
                             NDArrayIndex.all());
        this.test = all.get(NDArrayIndex.interval(trainSetSize, trainSetSize + testSetSize),
                            NDArrayIndex.all());
        this.validation = all.get(NDArrayIndex.interval(trainSetSize + testSetSize, rows),
                                  NDArrayIndex.all());
        
        all = null;


    }
}
