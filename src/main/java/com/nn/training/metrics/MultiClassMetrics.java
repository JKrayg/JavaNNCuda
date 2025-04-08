package com.nn.training.metrics;

import java.util.Arrays;

import org.ejml.simple.SimpleMatrix;

public class MultiClassMetrics extends Metrics{
    private double threshold;
    private SimpleMatrix confusionMatrix;

    public MultiClassMetrics() {
        this.threshold = 0.5;
    }

    public MultiClassMetrics(double threshold) {
        this.threshold = threshold;
    }

    public void getMetrics(SimpleMatrix pred, SimpleMatrix trueVals) {
        String dis = "Accuracy: " + accuracy(pred, trueVals) + "\n";
        dis += "Precision: ";
        for (double d: precision(pred, trueVals)) {
            dis += Double.toString(d) + ", ";
        }
        dis += "\n";
        // dis += "Precision: " + precision(pred, trueVals) + "\n";
        // dis += "Recall: " + recall(pred, trueVals) + "\n";
        System.out.println(dis);
    }

    public SimpleMatrix confusion(SimpleMatrix pred, SimpleMatrix trueVals) {
        double[][] preds = thresh(pred);
        int rows = pred.getNumRows();
        int cols = pred.getNumCols();
        SimpleMatrix cm = new SimpleMatrix(cols, cols);

        for (int i = 0; i < cols; i++) {
            SimpleMatrix currClass = new SimpleMatrix(1, cols);
            SimpleMatrix currClassPred = new SimpleMatrix(new SimpleMatrix(preds).getColumn(i));
            SimpleMatrix currClassTrue = trueVals.getColumn(i);
            // System.out.println(currClassPred);
            for (int j = 0; j < rows; j++) {
                double currPred = currClassPred.get(j);
                if (currPred == 1.0) {
                    for (int k = 0; k < cols; k++) {
                        if (trueVals.get(j, k) == 1.0) {
                            cm.set(i, k, cm.get(k) + 1.0);
                        }
                    }
                }
                    // else if (currClassPred.get(j) == 0.0) {
                    //     if (trueVals.get(j, k) == 1.0) {
                    //         cm.set(i, k, cm.get(k) + 1.0);
                    //     }
                    // }

            }
        }

        // for (int i = 0; i < preds.length; i++) {
        //     double label = trueVals.get(i);
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

        // SimpleMatrix confusionMatrix = new SimpleMatrix(new double[][]{{tp, fn}, {fp, tn}});
        this.confusionMatrix = cm;
        return cm;
    }

    public double accuracy(SimpleMatrix pred, SimpleMatrix trueVals) {
        int correct = 0;
        double[][] preds = thresh(pred);

        for (int i = 0; i < preds.length; i++) {
            for (int j = 0; j < preds[0].length; j++) {
                if (preds[i][j] == 1.0 && trueVals.get(i, j) == 1.0) {
                    correct += 1;
                }
            }
            
        }
        return ((double) correct) / ((double) preds.length);
    }


    public double[] precision(SimpleMatrix pred, SimpleMatrix trueVals) {
        // int correct = 0;
        // int wrong = 0;
        double[][] preds = thresh(pred);
        double[] classPrecisions = new double[preds[0].length];

        for (int i = 0; i < preds[0].length; i++) {
            SimpleMatrix currClassPred = new SimpleMatrix(new SimpleMatrix(preds).getColumn(i));
            SimpleMatrix currClassTrue = trueVals.getColumn(i);
            int tp = 0;
            int fp = 0;
            for (int j = 0; j < preds.length; j++) {
                if (currClassPred.get(j) == 1.0 && currClassTrue.get(j) == 1.0) {
                    tp += 1;
                } else if (currClassPred.get(j) == 1.0) {
                    fp += 1;
                }
            }

            double prec = ((double) tp) / (((double) tp) + ((double) fp));
            if (Double.isNaN(prec)) {
                classPrecisions[i] = 0.0;
            } else {
                classPrecisions[i] = prec;
            }
            for (int j = 0; j < preds[0].length; j++) {
                if (trueVals.get(i, j) == 1.0 && preds[i][j] == 1.0) {
                    tp += 1;
                } else if (preds[i][j] == 1.0) {
                    fp += 1;
                }
            }  
        }

        return classPrecisions;
    }

    public double recall(SimpleMatrix pred, SimpleMatrix trueVals) {
        return 0.0;
    }

    public double f1(SimpleMatrix pred, SimpleMatrix trueVals) {
        return 0.0;
    }

    public double[][] thresh(SimpleMatrix pred) {
        double[][] preds = new double[pred.getNumRows()][pred.getNumCols()];
        for (int i = 0; i < pred.getNumRows(); i++) {
            for (int j = 0; j < pred.getNumCols(); j++) {
                double highestProb = pred.getRow(i).elementMax();
                if (pred.get(i, j) == highestProb) {
                    preds[i][j] = 1.0;
                } else {
                    preds[i][j] = 0.0;
                }

            }
        }

        return preds;
    }
    
}
