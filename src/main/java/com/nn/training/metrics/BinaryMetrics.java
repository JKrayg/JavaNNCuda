package com.nn.training.metrics;

import org.ejml.simple.SimpleMatrix;

public class BinaryMetrics extends Metrics {
    private double threshold;
    private double tp;
    private double fp;
    private double tn;
    private double fn;
    private SimpleMatrix confusionMatrix;

    public BinaryMetrics() {
        this.threshold = 0.5;
    }

    public BinaryMetrics(double threshold) {
        this.threshold = threshold;
    }

    public void getMetrics(SimpleMatrix pred, SimpleMatrix trueVals) {
        System.out.println("Confusion Matrix: ");
        System.out.println(confusion(pred, trueVals));
        String dis = "Accuracy: " + accuracy(pred, trueVals) + "\n";
        dis += "Precision: " + precision(pred, trueVals) + "\n";
        dis += "Recall: " + recall(pred, trueVals) + "\n";
        dis += "F1 score: " + f1(pred, trueVals) + "\n";
        System.out.println(dis);
    }

    public SimpleMatrix confusion(SimpleMatrix pred, SimpleMatrix trueVals) {
        double[] preds = thresh(pred);
        double tp = 0.0;
        double fp = 0.0;
        double tn = 0.0;
        double fn = 0.0; 

        for (int i = 0; i < preds.length; i++) {
            double label = trueVals.get(i);
            if (preds[i] == 1.0) {
                if (label == 1.0) {
                    tp += 1.0;
                } else {
                    fp += 1.0;
                }
            } else {
                if (label == 1.0) {
                    fn += 1.0;
                } else {
                    tn += 1.0;
                }
            }
        }

        this.tp = tp;
        this.fp = fp;
        this.tn = tn;
        this.fn = fn;

        SimpleMatrix confusionMatrix = new SimpleMatrix(new double[][]{{tp, fn}, {fp, tn}});
        this.confusionMatrix = confusionMatrix;
        return confusionMatrix;
    }


    public double accuracy(SimpleMatrix pred, SimpleMatrix trueVals) {
        return (tp + tn) / pred.getNumRows();
    }


    public double precision(SimpleMatrix pred, SimpleMatrix trueVals) {
        double prec = tp / (tp + fp);
        if (Double.isNaN(prec)) {
            return 0.0;
        } else {
            return prec;
        }
    }


    public double recall(SimpleMatrix pred, SimpleMatrix trueVals) {
        double rec = tp / (tp + fn);
        if (Double.isNaN(rec)) {
            return 0.0;
        } else {
            return rec;
        }
    }


    public double f1(SimpleMatrix pred, SimpleMatrix trueVals) {
        double r = recall(pred, trueVals);
        double p = precision(pred, trueVals);
        if (Double.isNaN(r) || Double.isNaN(p)) {
            return 0.0;
        } else {
            return 2.0 / (1.0 / p + 1.0 / r);
        }
    }


    public double[] thresh(SimpleMatrix pred) {
        double[] preds = new double[pred.getNumRows()];
        for (int i = 0; i < preds.length; i++) {
            if (pred.get(i) > threshold) {
                preds[i] = 1.0;
            } else {
                preds[i] = 0.0;
            }
        }

        return preds;
    }
}
