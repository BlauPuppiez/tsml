package ml_6002b_coursework;

import java.util.Arrays;

/**
 * Empty class for Part 2.1 of the coursework.
 * Each contingency table is 'y coordinate' first then 'x coordinate'
 * e.g.
 * -----
 * |5|7|
 * |6|8|
 * -----
 * [1][0] yields 6; and [0][1] yields 7.
 */
public class AttributeMeasures {

    public static double log2(double num) {
        return Math.log(num) / Math.log(2);
    }

    /**
     * Information gain for contingency table.
     * Any values with 0 log 0 treated as zero. i.e. they are ignored.
     * As any additions of 0 do not change the value.
     */
    public static double measureInformationGain(int[][] contingencyTable) {
        int attValueCount = contingencyTable.length;
        int classCount = contingencyTable[0].length;

        int count = 0;
        int[] attTotals = new int[attValueCount];
        int[] classTotals = new int[classCount];
        for (int attIt = 0; attIt < attValueCount; attIt++) {
            for (int classIt = 0; classIt < classCount; classIt++) {
                count += contingencyTable[attIt][classIt];
                attTotals[attIt] += contingencyTable[attIt][classIt];
                classTotals[classIt] += contingencyTable[attIt][classIt];
            }
        }

        double hx = 0.0; // H(X)
        double sumY = 0.0;
        for (int attIt = 0; attIt < attValueCount; attIt++) {
            double globProb = (double)attTotals[attIt]/count; // Global probability
            hx -= globProb * log2(globProb);

            double currY = 0.0;
            for (int classIt = 0; classIt < classCount; classIt++) {
                double value = (double)contingencyTable[attIt][classIt]/attTotals[attIt];
                if (value != 0) {
                    currY += value * log2(value);
                }
            }
            sumY += globProb * currY; // Scale with instance count for each attribute
        }

        return hx + sumY;
    }

    /**
     * Information gain ratio for the contingency table.
     */
    public static double measureInformationGainRatio(int[][] contingencyTable) {
        int attValueCount = contingencyTable.length;
        int classCount = contingencyTable[0].length;
        int count = 0;
        int[] attTotals = new int[attValueCount];
        for (int attIt = 0; attIt < attValueCount; attIt++) {
            for (int classIt = 0; classIt < classCount; classIt++) {
                count += contingencyTable[attIt][classIt];
                attTotals[attIt] += contingencyTable[attIt][classIt];
            }
        }
        double informationGain = measureInformationGain(contingencyTable);

        double splitInformation = 0.0;
        for (int attIt = 0; attIt < attValueCount; attIt++) {
            double value = (double)attTotals[attIt]/count;
            splitInformation -= value * log2(value);
        }
        return informationGain/splitInformation;
    }

    /**
     * Gini measure for the contingency table.
     */
    public static double measureGini(int[][] contingencyTable) {
        int attValueCount = contingencyTable.length;
        int classCount = contingencyTable[0].length;
        int count = 0;
        int[] attTotals = new int[attValueCount];
        int[] classTotals = new int[classCount];
        for (int attIt = 0; attIt < attValueCount; attIt++) {
            for (int classIt = 0; classIt < classCount; classIt++) {
                count += contingencyTable[attIt][classIt];
                attTotals[attIt] += contingencyTable[attIt][classIt];
                classTotals[classIt] += contingencyTable[attIt][classIt];
            }
        }
        double ix = 1; // I(X)
        for (int classIt = 0; classIt < classCount; classIt++) {
            ix -= (double)classTotals[classIt]/count * classTotals[classIt]/count;
        }
        //System.out.println("I(X): " + ix); // Debug: Check I(X)

        double attSumm = 0.0; // Sum of all I(Y)
        for (int attIt = 0; attIt < attValueCount; attIt++) {
            double iy = 1.0; // I(Y)
            for (int classIt = 0; classIt < classCount; classIt++) {
                double value = (double)contingencyTable[attIt][classIt]/attTotals[attIt];
                iy -= value * value;
            }
            //System.out.println("I(Y): " + iy); // Debug: Check I(Y)

            double globProb = (double)attTotals[attIt]/count; // Global probability
            //System.out.println("SCALE: " + globProb); // Debug: Check scaling value
            attSumm += globProb * iy; // Scale with instance count for each attribute
            //System.out.println("SCALED I(Y): " + globProb * iy); // Debug: Check scaled I(Y) value
        }
        //System.out.println("SUM (IY): " + attSumm); // Debug: Check sum of I(Y)

        return ix - attSumm;
    }

    /**
     * Chi-squared statistic for the contingency table.
     */
    public static double measureChiSquared(int[][] contingencyTable) {
        int attValueCount = contingencyTable.length;
        int classCount = contingencyTable[0].length;
        int count = 0; // Total instance count
        int[] attTotals = new int[attValueCount];
        int[] classTotals = new int[classCount];
        for (int attIt = 0; attIt < attValueCount; attIt++) {
            for (int classIt = 0; classIt < classCount; classIt++) {
                count += contingencyTable[attIt][classIt];
                attTotals[attIt] += contingencyTable[attIt][classIt];
                classTotals[classIt] += contingencyTable[attIt][classIt];
            }
        }

        double chiSquared = 0.0;
        for (int attIt = 0; attIt < attValueCount; attIt++) {
            for (int classIt = 0; classIt < classCount; classIt++) {
//                System.out.println("Observed Value: " + contingencyTable[attIt][classIt]);
                double probability = (double)classTotals[classIt]/count;
//                System.out.println("Global Probability: " + probability); // Debug
                double expectedVal = attTotals[attIt] * probability;
//                System.out.println("Expected Value: " + expectedVal); // Debug
                chiSquared += Math.pow(contingencyTable[attIt][classIt] - expectedVal, 2) / expectedVal;
            }
        }
        return chiSquared;
    }

    /**
     * Main method.
     *
     * @param args the options for the attribute measure main
     */
    public static void main(String[] args) {
        /*
                         Islay Speyside
            Peaty: True    4      0
                   False   1      5
        */
        int[][] peatyContingencyTable = new int[][]{{4, 0}, {1, 5}};
        System.out.println(measureInformationGain(peatyContingencyTable));
        System.out.println(measureInformationGainRatio(peatyContingencyTable));
        System.out.println(measureGini(peatyContingencyTable));
        System.out.println(measureChiSquared(peatyContingencyTable));
    }

}
