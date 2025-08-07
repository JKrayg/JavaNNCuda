package com.nn.training.metrics;

import java.util.Arrays;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class MultiClassMetrics extends Metrics{
    private float threshold;
    private INDArray confusionMatrix;
    private float accuracy;

    public MultiClassMetrics() {
        this.threshold = 0.5f;
    }

    public MultiClassMetrics(float threshold) {
        this.threshold = threshold;
    }

    public void getMetrics(INDArray pred, INDArray target) {
        String dis = "Accuracy: " + accuracy + "\n";
        dis += "Precision: ";
        for (float d: precision(pred, target)) {
            dis += Float.toString(d) + ", ";
        }
        dis += "\n";
        // dis += "Precision: " + precision(pred, target) + "\n";
        // dis += "Recall: " + recall(pred, target) + "\n";
        System.out.println(dis);
    }

    public INDArray confusion(INDArray pred, INDArray target) {
        float[][] preds = thresh(pred);
        int rows = pred.rows();
        int cols = pred.columns();
        INDArray cm = Nd4j.create(cols, cols);

        for (int i = 0; i < cols; i++) {
            INDArray currClass = Nd4j.create(1, cols);
            INDArray currClassPred = Nd4j.create(preds).getColumn(i);
            INDArray currClassTrue = target.getColumn(i);
            // System.out.println(currClassPred);
            for (int j = 0; j < rows; j++) {
                float currPred = currClassPred.getFloat(j);
                if (currPred == 1.0) {
                    for (int k = 0; k < cols; k++) {
                        if (target.getFloat(j, k) == 1.0) {
                            cm.put(i, k, cm.getFloat(k) + 1.0);
                        }
                    }
                }
                    // else if (currClassPred.get(j) == 0.0) {
                    //     if (target.get(j, k) == 1.0) {
                    //         cm.set(i, k, cm.get(k) + 1.0);
                    //     }
                    // }

            }
        }

        // for (int i = 0; i < preds.length; i++) {
        //     float label = target.get(i);
        //     if (preds[i] == 1.0) {
        //         if (label == 1.0) {
        //             tp += 1.0;
        //         } else {
        //             fp += 1.0;
        //         }
        //     } else {
        //         if (label == 1.0) {
        //             fn += 1.0;
        //         } else {
        //             tn += 1.0;
        //         }
        //     }
        // }

        // INDArray confusionMatrix = Nd4j.create(new float[][]{{tp, fn}, {fp, tn}});
        this.confusionMatrix = cm;
        return cm;
    }

    public float accuracy(INDArray pred, INDArray target) {
        int correct = 0;
        float[][] preds = thresh(pred);

        for (int i = 0; i < preds.length; i++) {
            for (int j = 0; j < preds[0].length; j++) {
                if (preds[i][j] == 1.0 && target.getFloat(i, j) == 1.0) {
                    correct += 1;
                }
            }
            
        }
        return ((float) correct) / ((float) preds.length);
    }


    public float[] precision(INDArray pred, INDArray target) {
        // int correct = 0;
        // int wrong = 0;
        float[][] preds = thresh(pred);
        float[] classPrecisions = new float[preds[0].length];

        for (int i = 0; i < preds[0].length; i++) {
            INDArray currClassPred = Nd4j.create(preds).getColumn(i);
            INDArray currClassTrue = target.getColumn(i);
            int tp = 0;
            int fp = 0;
            for (int j = 0; j < preds.length; j++) {
                if (currClassPred.getFloat(j) == 1.0 && currClassTrue.getFloat(j) == 1.0) {
                    tp += 1;
                } else if (currClassPred.getFloat(j) == 1.0) {
                    fp += 1;
                }
            }

            float prec = ((float) tp) / (((float) tp) + ((float) fp));
            if (Float.isNaN(prec)) {
                classPrecisions[i] = 0.0f;
            } else {
                classPrecisions[i] = prec;
            }
            for (int j = 0; j < preds[0].length; j++) {
                if (target.getFloat(i, j) == 1.0 && preds[i][j] == 1.0) {
                    tp += 1;
                } else if (preds[i][j] == 1.0) {
                    fp += 1;
                }
            }  
        }

        return classPrecisions;
    }

    public float recall(INDArray pred, INDArray target) {
        return 0.0f;
    }

    public float f1(INDArray pred, INDArray target) {
        return 0.0f;
    }

    public float[][] thresh(INDArray pred) {
        int rows = pred.rows();
        int cols = pred.columns();
        float[][] preds = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float highestProb = pred.getRow(i).maxNumber().floatValue();
                if (pred.getFloat(i, j) == highestProb) {
                    preds[i][j] = 1.0f;
                } else {
                    preds[i][j] = 0.0f;
                }

            }
        }

        return preds;
    }
    
}
