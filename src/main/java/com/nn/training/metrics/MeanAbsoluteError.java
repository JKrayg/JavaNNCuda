package com.nn.training.metrics;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MeanAbsoluteError extends Metrics {
    
    public void getMetrics(INDArray pred, INDArray trueVals) {
        INDArray dif = pred.sub(trueVals);
        float mae = Transforms.abs(dif).sumNumber().floatValue() / trueVals.length();
        System.out.println("MAE: " + mae);
        // return dif.mul(dif).sumNumber().floatValue() / labels.length();
    }
}
