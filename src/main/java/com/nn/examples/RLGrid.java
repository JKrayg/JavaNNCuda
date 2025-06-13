package com.nn.examples;

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
import org.nd4j.linalg.ops.transforms.Transforms;

import com.nn.activation.ReLU;
import com.nn.activation.Softmax;
import com.nn.components.Layer;
import com.nn.components.NeuralNet;
import com.nn.layers.Dense;
import com.nn.layers.Output;
import com.nn.utils.MathUtils;

public class RLGrid {
    private static INDArray initialState;
    private static INDArray currentState;
    private static INDArray oldStates;
    private static INDArray actions;
    private static INDArray rewards;
    private static INDArray oldPolicyProbs;
    private static INDArray valueEstimates;
    private static int[] posAgent;
    private static int[] posGoal;
    private static int[] prevPos;
    private static int[][] posObstacles;
    private static int numEpisodes;
    private static int numObstacles;
    private static boolean done = false;
    private static boolean goal = false;
    private static NeuralNet policyNetwork;
    private static NeuralNet valueNetwork;
    private static float discountFactor = 0.99f;
    private static float gaeDecay = 0.95f;
    private static int step = 0;
    private static int maxSteps;


    public static INDArray initializeEnvironment(int r, int c, int numAgents, int numObst) {
        numObstacles = numObst;
        // maxSteps = c * 2;
        posAgent = new int[]{r / 2, 0};
        posGoal = new int[]{r / 2, c - 1};
        posObstacles = new int[][]{{r/2, c/2}, {r/2+1, c/2}, {r/2-1, c/2}};
        // oldStates = Nd4j.create(0, 3, r, c);
        // rewards = Nd4j.create(maxSteps);
        INDArray grid = Nd4j.create(3, r, c);

        grid.putScalar(new int[]{0, posAgent[0], posAgent[1]}, 1);
        grid.putScalar(new int[]{1, posGoal[0], posGoal[1]}, 2);

        if (numObstacles != -1) {
            int[][] positions = new int[r*c - 2][2];
            int idx = 0;

            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    int[] currPos = new int[]{i, j};
                    if (!Arrays.equals(currPos, posAgent) && !Arrays.equals(currPos, posGoal)) {
                        positions[idx] = new int[]{i, j};
                        idx++;
                    }
                }
            }

            INDArray pos = Nd4j.create(positions);
            Nd4j.shuffle(pos, 1);

            for (int i = 0; i < numObstacles; i++) {
                int[] obstPos = pos.get(NDArrayIndex.point(numAgents + 1 + i)).toIntVector();
                int[] op = new int[]{2, obstPos[0], obstPos[1]};
                grid.putScalar(op, 3);
            }
        } else {
            for (int i = 0; i < posObstacles.length; i++) {
                grid.putScalar(new int[]{2, posObstacles[i][0], posObstacles[i][1]}, 3);
            }
        }

        initialState = grid;
        currentState = grid;

        return grid;
    }

    public static float euc(int[] a, int[] b) {
        return (float) Math.sqrt(Math.pow((b[0] - a[0]), 2) + Math.pow((b[1] - a[1]), 2));
    }


    public static float reward(int[] currPos) {
        float reward = 0;

        float currDist = euc(currPos, posGoal);
        float prevDist;
        if (prevPos == null) {
            prevDist = euc(posAgent, posGoal);
        } else {
            prevDist = euc(prevPos, posGoal);
        }

        if (currDist < prevDist) {
            reward += 0.1f;
        } else if (currDist > prevDist) {
            reward += -0.05f;
        } else {
            reward += 0;
        }

        if (Arrays.equals(currPos, posGoal)) {
            reward += 1.0f;
            goal = true;
            done = true;
        } 
        
        if (posObstacles != null) {
            for (int i = 0; i < posObstacles.length; i++) {
                if (Arrays.equals(currPos, posObstacles[i])) {
                    reward += -0.75f;
                    done = true;
                }
            }
        }

        if (currPos[0] < 0 || currPos[1] < 0 || currPos[0] > oldStates.shape()[2] - 1 || currPos[1] > oldStates.shape()[3] - 1) {
            reward += -0.1f;
            posAgent = prevPos;
        }


        if (step == maxSteps - 1) {
            reward += -0.1f;
            done = true;
        }

        if (rewards == null) {
            rewards = Nd4j.create(new float[]{reward});
        } else {
            rewards = Nd4j.concat(0, Nd4j.create(new float[]{reward}), rewards);
        }

        prevPos = currPos.clone();

        return reward;
    }


    public static void reset() {
        currentState = initialState.dup();
    }


    public static void policyGradient() {

    }


    public static int[] up(int[] pos) {
        System.out.println("up");
        pos[0] -= 1;
        return pos;
    }


    public static int[] down(int[] pos) {
        System.out.println("down");
        pos[0] += 1;
        return pos;
    }


    public static int[] left(int[] pos) {
        System.out.println("left");
        pos[1] -= 1;
        return pos;
    }


    public static int[] right(int[] pos) {
        System.out.println("right");
        pos[1] += 1;
        return pos;
    }

    public static boolean isInBounds(int[] nextPos, long[] envShape) {
        if (nextPos[0] >= 0 && nextPos[1] >= 0 && nextPos[0] <= envShape[1] - 1 && nextPos[1] <= envShape[2] - 1) {
            return true;
        }

        return false;
    }


    public static int[] step(INDArray observation) {
        System.out.println(observation);
        int[] currPos = posAgent.clone();

        // policy network - forward pass only
        policyNetwork.forwardPass(observation.reshape(1, -1));
        Output pprobs = (Output) policyNetwork.getLayers().get(policyNetwork.getLayers().size() - 1);
        double[] probs = pprobs.getActivations().toDoubleVector();

        // value network - forward pass only
        valueNetwork.forwardPass(observation.reshape(1, -1));
        Output value = (Output) valueNetwork.getLayers().get(valueNetwork.getLayers().size() - 1);
        float v = value.getActivations().getFloat();

        Random rand = new Random();
        double r = rand.nextDouble();
        double cumulative = 0.0;
        int selectedMove = -1;

        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r < cumulative) {
                selectedMove = i;
                break;
            }
            selectedMove = probs.length - 1;
        }

        if (actions == null) {
            actions = Nd4j.create(new float[]{selectedMove});
        } else {
            actions = Nd4j.concat(0, Nd4j.create(new float[]{selectedMove}), actions);
        }


        if (oldPolicyProbs == null) {
            oldPolicyProbs = Nd4j.create(new float[]{(float)Math.log(probs[selectedMove])});
        } else {
            oldPolicyProbs = Nd4j.concat(0, Nd4j.create(new float[]{(float)Math.log(probs[selectedMove])}), oldPolicyProbs);
        }


        if (valueEstimates == null) {
            valueEstimates = Nd4j.create(new float[]{v});
        } else {
            valueEstimates = Nd4j.concat(0, Nd4j.create(new float[]{v}), valueEstimates);
        }

        if (oldStates == null) {
            oldStates = observation.reshape(1, observation.shape()[0], observation.shape()[1], observation.shape()[2]);
        } else {
            oldStates = Nd4j.concat(0, observation.reshape(1, observation.shape()[0], observation.shape()[1], observation.shape()[2]), oldStates);
        }


        int[] nextPos = getNextPos(posAgent, selectedMove);



        if (isInBounds(nextPos, observation.shape())) {
            observation.putScalar(new int[]{0, currPos[0], currPos[1]}, 0);
            observation.putScalar(new int[]{0, nextPos[0], nextPos[1]}, 1);
        }
        
        System.out.println(observation);
        posAgent = nextPos;
        prevPos = currPos;

        return nextPos;

    }

    public static INDArray normalize(INDArray advantages) {
        MathUtils maths = new MathUtils();

        float e = 1e-8f;
        long numAdv = advantages.length();
        float mean = (advantages.sumNumber().floatValue() / numAdv);
        advantages.subi(mean).divi(maths.std(advantages) + e);
        // int cols = advantages.columns();
        // int rows = advantages.rows();
        // for (int i = 0; i < cols; i++) {
        //     INDArray col = advantages.getColumn(i);
        //     float mean = (col.sumNumber().floatValue() / rows);
        //     float std = maths.std(col);
        //     for (int j = 0; j < rows; j++) {
        //         advantages.putScalar(j, i, (advantages.getFloat(j, i) - mean) / std);
        //     }
        // }

        return advantages;
    }

    public static int[] getNextPos(int[] pos, int move) {
        if (move == 0) {
            return up(posAgent);
        } else if (move == 1) {
            return down(posAgent);
        } else if (move == 2) {
            return left(posAgent);
        } else if (move == 3) {
            return right(posAgent);
        } else {
            return posAgent;
        }
    }

    // public static INDArray gae() {

    // }


    public static void run(int numEpisodes, int ms, INDArray observation) {
        maxSteps = ms;
        for (int i = 0; i < numEpisodes; i++) {
            while (done == false) {
                int[] nextPos = step(observation);
                reward(nextPos);
                if (done == false) {
                    step++;
                }

            }

            INDArray tdErrors = Nd4j.create(step);
            INDArray qVals = Nd4j.create(step);
            INDArray adv = Nd4j.create(step);

            for (int j = 0; j < step; j++) {
                tdErrors.put(j, rewards.get(NDArrayIndex.point(j))
                    .add(valueEstimates.get(NDArrayIndex.point(j + 1)).mul(discountFactor))
                    .sub(valueEstimates.get(NDArrayIndex.point(j))));

                qVals.put(j, rewards.get(NDArrayIndex.point(j))
                    .add(valueEstimates.get(NDArrayIndex.point(j + 1)).mul(discountFactor)));
            }

            adv.put(step - 1, tdErrors.get(NDArrayIndex.point(step - 1)));
            for (int k = step - 2; k >= 0; k--) {
                System.out.print(tdErrors.getFloat(k));
                System.out.print(" + " + (gaeDecay*discountFactor));
                System.out.print(" * " + adv.getScalar(k + 1));
                System.out.println(" = " + (tdErrors.getFloat(k) + (gaeDecay*discountFactor) * adv.getFloat(k + 1)));
                adv.put(k, tdErrors.get(NDArrayIndex.point(k)).add(adv.getScalar(k + 1).mul(gaeDecay*discountFactor)));
            }

            policyNetwork.forwardPass(oldStates.reshape(step + 1, -1));
            INDArray polProbs = policyNetwork.getLayers().get(policyNetwork.getLayers().size() - 1).getActivations();
            Softmax softmax = new Softmax();
            INDArray probDist = softmax.execute(polProbs);

            INDArray logProb = Nd4j.create(actions.shape());

            for (int h = 0; h < logProb.length(); h++) {
                logProb.put(h, probDist.get(NDArrayIndex.point(h), NDArrayIndex.point(actions.getInt(h))));
            }


            System.out.println(Arrays.toString(actions.shape()));
            System.out.println("actions: " + actions);
            // System.out.println(Arrays.toString(oldPolicyProbs.shape()));
            // System.out.println("policy probs: " + oldPolicyProbs);
            // System.out.println("policy probs exp: " + Transforms.exp(oldPolicyProbs));
            // System.out.println(Arrays.toString(valueEstimates.shape()));
            // System.out.println("value estimates: " + valueEstimates);
            // System.out.println(Arrays.toString(rewards.shape()));
            // System.out.println("rewards: " + rewards);
            // System.out.println(Arrays.toString(tdErrors.shape()));
            // System.out.println("tderror: " + tdErrors);
            // System.out.println(Arrays.toString(qVals.shape()));
            // System.out.println("qVals: " + qVals);
            // System.out.println(Arrays.toString(adv.shape()));
            // System.out.println("advantage: " + adv);
            // System.out.println(Arrays.toString(normalize(adv).shape()));
            // System.out.println("normalized advantage: " + normalize(adv));
            // System.out.println("old states shape: " + Arrays.toString(oldStates.shape()));
            // System.out.println(Arrays.toString(polProbs.shape()));
            // System.out.println("old states policy probs: " + polProbs);
            // System.out.println(Arrays.toString(probDist.shape()));
            System.out.println("probability dist (after softmax): " + probDist);
            System.out.println(Arrays.toString(logProb.shape()));
            System.out.println("logProb: " + logProb);

            if (goal == true) {
                done = true;
                System.out.println("GGGOOOOOOOOOOOOOOOOOOOOOOOOAAAAAAAAAAAAAAAAALLLLLLLLLLLLLLLLLLLLLL!!!!!!!!!!!!!!");
                break;
            }
        }
        
        
    }


    public static void main(String[] args) {

        INDArray grid = initializeEnvironment(5, 5, 1, -1);

        NeuralNet policy = new NeuralNet();
        Dense d1 = new Dense(64, new ReLU(), (int)initialState.reshape(-1).length());
        Dense d2 = new Dense(64, new ReLU());
        Output out = new Output(4, new Softmax());

        policy.addLayer(d1);
        policy.addLayer(d2);
        policy.addLayer(out);

        for (Layer l: policy.getLayers()) {
                l.initLayer(l.getPrev(), 1);
        }

        policyNetwork = policy;

        NeuralNet value = new NeuralNet();
        Dense dv1 = new Dense(64, new ReLU(), (int)initialState.reshape(-1).length());
        Dense dv2 = new Dense(64, new ReLU());
        Output outv = new Output(1);

        value.addLayer(dv1);
        value.addLayer(dv2);
        value.addLayer(outv);

        for (Layer l: value.getLayers()) {
                l.initLayer(l.getPrev(), 1);
        }

        valueNetwork = value;

        run(1, 20, grid);
        
    }
}
