package com.nn.examples;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import com.nn.activation.ReLU;
import com.nn.activation.Softmax;
import com.nn.components.Layer;
import com.nn.components.NeuralNet;
import com.nn.layers.Dense;
import com.nn.layers.Output;

public class RLGrid {
    private static INDArray initialState;
    private static INDArray currentState;
    private static INDArray oldStates;
    private static INDArray actions = Nd4j.createUninitialized();
    private static INDArray rewards;
    private static INDArray oldPolicyProbs = Nd4j.createUninitialized();
    private static INDArray valueEstimates = Nd4j.createUninitialized();
    private static int[] posAgent;
    private static int[] posGoal;
    private static int[] prevPos;
    private static int[][] posObstacles;
    private static int numEpisodes;
    private static int numObstacles;
    private static boolean done = false;
    private static NeuralNet policyNetwork;
    private static NeuralNet valueNetwork;
    
    private static int step = 0;
    private static int maxSteps;

    public static INDArray initializeEnvironment(int r, int c, int numAgents, int numObst) {
        numObstacles = numObst;
        maxSteps = c * 2;
        posAgent = new int[]{r / 2, 0};
        posGoal = new int[]{r / 2, c - 1};
        posObstacles = new int[][]{{r/2, c/2}, {r/2+1, c/2}, {r/2-1, c/2}};
        oldStates = Nd4j.create(0, 3, r, c);
        rewards = Nd4j.create(maxSteps);
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
        System.out.println(Arrays.toString(a));
        System.out.println(Arrays.toString(b));
        return (float) Math.sqrt(Math.pow((b[0] - a[0]), 2) + Math.pow((b[1] - a[1]), 2));
    }


    public static float reward(int[] currPos) {
        System.out.println("curr: " + Arrays.toString(currPos));
        float reward = 0;

        float currDist = euc(currPos, posGoal);
        float prevDist;
        if (prevPos == null) {
            prevDist = euc(posAgent, posGoal);
        } else {
            prevDist = euc(prevPos, posGoal);
        }

        System.out.println("prev dist: " + prevDist);
        System.out.println("curr dist: " + currDist);

        if (currDist < prevDist) {
            reward += 0.1f;
        } else if (currDist > prevDist) {
            reward += -0.05f;
        } else {
            reward += 0;
        }

        if (Arrays.equals(currPos, posGoal)) {
            reward += 1.0f;
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

        rewards.putScalar(step, reward);
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
        // System.out.println(Arrays.toString(observation.get(NDArrayIndex.point(0)).argMax().shape()));
        // int[] currPos = Where(observation.get(NDArrayIndex.point(0)).eq(1));
        // System.out.println("curr position: " + Arrays.toString(posAgent));
        int[] currPos = posAgent.clone();

        // policy network - forward pass only
        policyNetwork.forwardPass(observation.reshape(1, -1));
        Output pprobs = (Output) policyNetwork.getLayers().get(policyNetwork.getLayers().size() - 1);
        double[] probs = pprobs.getActivations().toDoubleVector();

        // value
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

        Nd4j.concat(0, actions, Nd4j.create(new int[]{selectedMove}));
        Nd4j.concat(0, oldPolicyProbs, Nd4j.create(new float[]{(float)Math.log(probs[selectedMove])}));
        Nd4j.concat(0, valueEstimates, Nd4j.create(new float[]{v}));
        Nd4j.concat(0, oldStates, observation.reshape(1, observation.shape()[0], observation.shape()[1], observation.shape()[2]));


        int[] nextPos;

        if (selectedMove == 0) {
            nextPos = up(posAgent);
        } else if (selectedMove == 1) {
            nextPos = down(posAgent);
        } else if (selectedMove == 2) {
            nextPos = left(posAgent);
        } else if (selectedMove == 3) {
            nextPos = right(posAgent);
        } else {
            nextPos = posAgent;
        }



        if (isInBounds(nextPos, observation.shape())) {
            observation.putScalar(new int[]{0, currPos[0], currPos[1]}, 0);
            observation.putScalar(new int[]{0, nextPos[0], nextPos[1]}, 1);
        }
        
        System.out.println(observation);
        posAgent = nextPos;
        prevPos = currPos;

        return nextPos;

    }


    public static void run(int numEpisodes, INDArray observation) {
        for (int i = 0; i < numEpisodes; i++) {
            while (done == false) {
                int[] nextPos = step(observation);
                float re = reward(nextPos);
                System.out.println("reward: " + re);
                System.out.println(rewards);
                step++;
            }
        }

        System.out.println(Arrays.toString(oldStates.shape()));
        
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

        run(1, grid);
        
    }
}
