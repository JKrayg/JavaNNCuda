package com.nn.training.metrics;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MeanAbsoluteError extends Metrics {
    
    public void getMetrics(INDArray pred, INDArray target) {
        INDArray dif = pred.sub(target);
        float mae = Transforms.abs(dif).sumNumber().floatValue() / target.length();
        System.out.println("MAE: " + mae);
        // return dif.mul(dif).sumNumber().floatValue() / preds.length();
    }

    @Override
    public float accuracy(INDArray pred, INDArray target) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'accuracy'");
    }
}
