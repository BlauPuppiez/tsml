package ml_6002b_coursework;

import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.SimpleCart;
import weka.core.Instance;
import weka.core.Instances;

import java.math.BigDecimal;
import java.math.RoundingMode;

public class Evaluation {

    /**
     * Rounds to 2 d.p. or whatever is set
     * From: <a href="https://stackoverflow.com/questions/2808535/round-a-double-to-2-decimal-places">Stack Overflow Round to 2 dp</a>
     * @param value to round
     * @return rounded value
     */
    public static double round(double value) {
        BigDecimal bd = BigDecimal.valueOf(value);
        bd = bd.setScale(0, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }

    public static void main(String[] args) throws Exception {
        /* Discrete + Continuous Dataset loading
        String[] discreteDatasets = DatasetLists.nominalAttributeProblems;
        String[] continuousDatasets = DatasetLists.continuousAttributeProblems;
        int discreteCount = discreteDatasets.length;
        int continuousCount = continuousDatasets.length;

        Instances[] discreteData = new Instances[discreteCount];
        Instances[] continuousData = new Instances[continuousCount];

        for (int i = 0; i < discreteCount; i++) {
            String filename = "UCI Discrete\\" + discreteDatasets[i] + "\\" + discreteDatasets[i] + ".arff";
            discreteData[i] = WekaTools.loadLocalClassificationData(filename);
        }
        for (int i = 0; i < continuousCount; i++) {
            String filename = "UCI Continuous\\" + continuousDatasets[i] + "\\" + continuousDatasets[i] + ".arff";
            continuousData[i] = WekaTools.loadLocalClassificationData(filename);
        }
        //*/

        Instances insectWingbeatTrain = WekaTools.loadLocalClassificationData("InsectWingbeatTRAIN.arff");
        Instances insectWingbeatTest = WekaTools.loadLocalClassificationData("InsectWingbeatTEST.arff");

        // Summary of datasets:
        ///*
        System.out.println("Summary of Datasets");
        /*
        System.out.println("Discrete Datasets " + discreteCount);
        for (int discreteIt = 0; discreteIt < discreteCount; discreteIt++) {
            if (discreteIt != 1) {
//                System.out.println(discreteDatasets[discreteIt]);
//                System.out.println(discreteData[discreteIt].numAttributes()); // Attribute Count
//                System.out.println(discreteData[discreteIt].size()); // Instance Count
//                System.out.println(discreteData[discreteIt].numClasses()); // Classes
                // Distribution
                int[] classDistributionCount = new int[discreteData[discreteIt].numClasses()];
                for (Instance instance : discreteData[discreteIt]) {
                    classDistributionCount[(int)instance.classValue()]++;
                }
                int instanceCount = discreteData[discreteIt].size();
                double[] classDistribution = new double[discreteData[discreteIt].numClasses()];
                String string = "";
                for (int i = 0; i < classDistribution.length; i++) {
                    classDistribution[i] = (double)classDistributionCount[i] / instanceCount;
//                    string += classDistributionCount[i] + ","; // Append count
                    string += round(classDistribution[i] * 100) + ","; // Append proportions
                }
                string = string.substring(0, string.length()-1);
                //System.out.println(string);
            }
        }
        System.out.println("Continuous Datasets " + continuousCount);
        for (int continuousIt = 0; continuousIt < continuousCount; continuousIt++) {
//            System.out.println(continuousDatasets[continuousIt]);
//            System.out.println(continuousData[continuousIt].numAttributes()); // Attribute Count
//            System.out.println(continuousData[continuousIt].size()); // Instance Count
//            System.out.println(continuousData[continuousIt].numClasses()); // Classes
            // Distribution
            int[] classDistributionCount = new int[continuousData[continuousIt].numClasses()];
            for (Instance instance : continuousData[continuousIt]) {
                classDistributionCount[(int)instance.classValue()]++;
            }
            int instanceCount = continuousData[continuousIt].size();
            double[] classDistribution = new double[continuousData[continuousIt].numClasses()];
            String string = "";
            for (int i = 0; i < classDistribution.length; i++) {
                classDistribution[i] = (double)classDistributionCount[i] / instanceCount;
//                string += classDistributionCount[i] + ","; // Append count
                string += round(classDistribution[i] * 100) + ","; // Append proportions
            }
            string = string.substring(0, string.length()-1);
//            System.out.println(string);
        } //*/

        // Specified Dataset:
        System.out.println("InsectWingbeatTrain Summary");
        System.out.println(insectWingbeatTrain.numAttributes()); // Attribute Count
        System.out.println(insectWingbeatTrain.size()); // Instance Count
        System.out.println(insectWingbeatTrain.numClasses()); // Classes
        // Distribution
        int[] classDistributionCount = new int[insectWingbeatTrain.numClasses()];
        for (Instance instance : insectWingbeatTrain) {
            classDistributionCount[(int)instance.classValue()]++;
        }
        int instanceCount = insectWingbeatTrain.size();
        double[] classDistribution = new double[insectWingbeatTrain.numClasses()];
        String countString = "";
        String disString = "";
        for (int i = 0; i < classDistribution.length; i++) {
            classDistribution[i] = (double)classDistributionCount[i] / instanceCount;
            countString += classDistributionCount[i] + ","; // Append count
            disString += round(classDistribution[i] * 100) + ","; // Append proportions
        }
        countString = countString.substring(0, countString.length()-1);
        disString = disString.substring(0, disString.length()-1);
        System.out.println(countString);
        System.out.println(disString);
        System.out.println("InsectWingbeatTest Summary");
        System.out.println(insectWingbeatTest.numAttributes()); // Attribute Count
        System.out.println(insectWingbeatTest.size()); // Instance Count
        System.out.println(insectWingbeatTest.numClasses()); // Classes
        // Distribution
        classDistributionCount = new int[insectWingbeatTest.numClasses()];
        for (Instance instance : insectWingbeatTest) {
            classDistributionCount[(int)instance.classValue()]++;
        }
        instanceCount = insectWingbeatTest.size();
        classDistribution = new double[insectWingbeatTest.numClasses()];
        countString = "";
        disString = "";
        for (int i = 0; i < classDistribution.length; i++) {
            classDistribution[i] = (double)classDistributionCount[i] / instanceCount;
            countString += classDistributionCount[i] + ","; // Append count
            disString += round(classDistribution[i] * 100) + ","; // Append proportions
        }
        countString = countString.substring(0, countString.length()-1);
        disString = disString.substring(0, disString.length()-1);
        System.out.println(countString);
        System.out.println(disString);
        //*/ End of Summary

        // Compare with Weka ID3 and J48 classifiers
        /*
        System.out.println("Evaluation of Discrete Datasets");
        for (int discreteIt = 0; discreteIt < discreteCount; discreteIt++) {
            if (discreteIt != 1) {
                Instances dataset = discreteData[discreteIt];
                Instances[] dataSplit = WekaTools.splitData(dataset, 0.8);

                System.out.println("Evaluation of " + discreteDatasets[discreteIt] + ":");
                // TODO courseworkTree accuracy... etc.
//                CourseworkTree courseworkTree = new CourseworkTree();
//                courseworkTree.buildClassifier(dataSplit[0]);
//                double ctAcc = WekaTools.accuracy(courseworkTree, dataSplit[1]);
//                System.out.println("CT Accuracy: " + ctAcc);

//                Id3 id3 = new Id3();
//                id3.buildClassifier(dataSplit[0]);
//                double id3Acc = WekaTools.accuracy(id3, dataSplit[1]);
//                System.out.println("ID3 Accuracy: " + id3Acc);
//
//                J48 j48 = new J48();
//                j48.buildClassifier(dataSplit[0]);
//                double j48Acc = WekaTools.accuracy(j48, dataSplit[1]);
//                System.out.println("J48 Accuracy: " + j48Acc);
            }
        } //*/

        // And for Continuous Datasets
        /*
        System.out.println("Evaluation of Continuous Datasets");
        for (int continuousIt = 0; continuousIt < continuousCount; continuousIt++) {
            if (continuousIt != 1) {
                Instances dataset = continuousData[continuousIt];
                Instances[] dataSplit = WekaTools.splitData(dataset, 0.8);

                System.out.println("Evaluation of " + continuousDatasets[continuousIt] + ":");
                // TODO courseworkTree accuracy... etc.
//                CourseworkTree courseworkTree = new CourseworkTree();
//                courseworkTree.buildClassifier(dataSplit[0]);
//                double ctAcc = WekaTools.accuracy(courseworkTree, dataSplit[1]);
//                System.out.println("CT Accuracy: " + ctAcc);
                System.out.println("CourseworkTree");
                System.out.println("SimpleCART");
                System.out.println("J48");

//                SimpleCart simpleCart = new SimpleCart();
//                simpleCart.buildClassifier(dataSplit[0]);
//                double cartAcc = WekaTools.accuracy(simpleCart, dataSplit[1]);
//                System.out.println("CART Accuracy: " + cartAcc);
//
//                J48 j48 = new J48();
//                j48.buildClassifier(dataSplit[0]);
//                double j48Acc = WekaTools.accuracy(j48, dataSplit[1]);
//                System.out.println("J48 Accuracy: " + j48Acc);
            }
        } //*/
    }
}
