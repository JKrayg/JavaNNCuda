package com.nn.training.metrics;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class BinaryMetrics extends Metrics {
    private float threshold;
    private float tp;
    private float fp;
    private float tn;
    private float fn;
    private INDArray confusionMatrix;

    public BinaryMetrics() {
        this.threshold = 0.5f;
    }

    public BinaryMetrics(float threshold) {
        this.threshold = threshold;
    }

    public void getMetrics(INDArray pred, INDArray trueVals) {
        System.out.println("Confusion Matrix: ");
        System.out.println(confusion(pred, trueVals));
        String dis = "Accuracy: " + accuracy(pred, trueVals) + "\n";
        dis += "Precision: " + precision(pred, trueVals) + "\n";
        dis += "Recall: " + recall(pred, trueVals) + "\n";
        dis += "F1 score: " + f1(pred, trueVals) + "\n";
        System.out.println(dis);
    }

    public INDArray confusion(INDArray pred, INDArray trueVals) {
        INDArray preds = thresh(pred);
        INDArray confMatrix = Nd4j.create(2, 2);
        float tp = 0.0f;
        float fp = 0.0f;
        float tn = 0.0f;
        float fn = 0.0f; 

        for (int i = 0; i < preds.rows(); i++) {
            float label = trueVals.getFloat(i);
            if (preds.getDouble(i) == 1.0) {
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

        INDArray confusionMatrix = Nd4j.create(new float[][]{{tp, fn}, {fp, tn}});
        this.confusionMatrix = confusionMatrix;
        return confusionMatrix;
    }


    public float accuracy(INDArray pred, INDArray trueVals) {
        return (tp + tn) / pred.rows();
    }


    public float precision(INDArray pred, INDArray trueVals) {
        float prec = tp / (tp + fp);
        if (Float.isNaN(prec)) {
            return 0.0f;
        } else {
            return prec;
        }
    }


    public float recall(INDArray pred, INDArray trueVals) {
        float rec = tp / (tp + fn);
        if (Float.isNaN(rec)) {
            return 0.0f;
        } else {
            return rec;
        }
    }


    public float f1(INDArray pred, INDArray trueVals) {
        float r = recall(pred, trueVals);
        float p = precision(pred, trueVals);
        if (Float.isNaN(r) || Float.isNaN(p)) {
            return 0.0f;
        } else {
            return (float) (2.0 / (1.0 / p + 1.0 / r));
        }
    }


    public INDArray thresh(INDArray pred) {
        INDArray t = pred.gt(threshold).castTo(DataType.FLOAT);
        // float[] preds = new float[pred.rows()];
        // for (int i = 0; i < preds.length; i++) {
        //     if (pred.getFloat(i) > threshold) {
        //         preds[i] = 1.0f;
        //     } else {
        //         preds[i] = 0.0f;
        //     }
        // }

        return t;
    }
}
