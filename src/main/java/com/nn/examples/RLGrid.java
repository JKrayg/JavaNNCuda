package com.nn.examples;

import java.util.ArrayList;
import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class RLGrid {
    public static INDArray generateMap(int x, int y, int numAgents, int numGoals, int numObstacles) {
        Random r = new Random();
        INDArray grid = Nd4j.create(3, x, y);
        INDArray agentPosition = Nd4j.create(2, numAgents);
        INDArray goalPosition = Nd4j.create(2, numGoals);
        INDArray obstaclePosition = Nd4j.create(2, numObstacles);
        ArrayList<int[]> used = new ArrayList<>();

        for (int i = 0; i < numAgents; i++) {
            int[] rPos = new int[]{r.nextInt(x), r.nextInt(y)};
            while (used.contains(rPos)) {
                rPos = new int[]{r.nextInt(x), r.nextInt(y)};
            }

            int[] p = new int[]{0, rPos[0], rPos[1]};
            grid.putScalar(p, 1);
        }

        for (int i = 0; i < numGoals; i++) {
            int[] rPos = new int[]{r.nextInt(x), r.nextInt(y)};
            while (used.contains(rPos)) {
                rPos = new int[]{r.nextInt(x), r.nextInt(y)};
            }
            int[] p = new int[]{1, rPos[0], rPos[1]};
            grid.putScalar(p, 1);
        }

        for (int i = 0; i < numObstacles; i++) {
            int[] rPos = new int[]{r.nextInt(x), r.nextInt(y)};
            while (used.contains(rPos)) {
                rPos = new int[]{r.nextInt(x), r.nextInt(y)};
            }

            int[] p = new int[]{2, rPos[0], rPos[1]};
            grid.putScalar(p, 1);
        }


        return grid;
    }
    public static void main(String[] args) {

        System.out.println(generateMap(5, 5, 2, 2, 2));
        
    }
}
