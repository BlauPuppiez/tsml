package ml_6002b_coursework;

import org.checkerframework.checker.units.qual.C;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.*;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.*;
import weka.core.Instance;
import weka.core.Instances;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Arrays;

public class Evaluation {

    private static void SummariseData(Instances data) {
        System.out.println(data.numAttributes() - 1); // Attribute Count
        System.out.println(data.size()); // Instance Count
        System.out.println(data.numClasses()); // Classes
        // Distribution
        int[] classDistributionCount = new int[data.numClasses()];
        for (Instance instance : data) {
            classDistributionCount[(int)instance.classValue()]++;
        }
        int instanceCount = data.size();
        double[] classDistribution = new double[data.numClasses()];
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
    }

    /**
     * Rounds to 2 d.p. or whatever is set
     * From: <a href="https://stackoverflow.com/questions/2808535/round-a-double-to-2-decimal-places">Stack Overflow Round to 2 dp</a>
     * @param value to round
     * @return rounded value
     */
    private static double round(double value) {
        BigDecimal bd = BigDecimal.valueOf(value);
        bd = bd.setScale(0, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }

    public static void main(String[] args) throws Exception {
        ///* Discrete + Continuous Dataset loading
        String[] discreteDatasetNames = DatasetLists.nominalAttributeProblems;
        String[] continuousDatasetNames = DatasetLists.continuousAttributeProblems;
        int discreteCount = discreteDatasetNames.length;
        int continuousCount = continuousDatasetNames.length;

        Instances[] discreteData = new Instances[discreteCount];
        Instances[] continuousData = new Instances[continuousCount];

        for (int i = 0; i < discreteCount; i++) {
            String filename = "UCI Discrete\\" + discreteDatasetNames[i] + "\\" + discreteDatasetNames[i] + ".arff";
            discreteData[i] = WekaTools.loadLocalClassificationData(filename);
        }
        for (int i = 0; i < continuousCount; i++) {
            String filename = "UCI Continuous\\" + continuousDatasetNames[i] + "\\" + continuousDatasetNames[i] + ".arff";
            continuousData[i] = WekaTools.loadLocalClassificationData(filename);
        }
        System.out.println("Loaded UCI data");
        //*/

        ///* Case Study Datasets:
        Instances insectSoundTrain = WekaTools.loadLocalClassificationData("InsectSoundTRAIN.arff");
        Instances insectSoundTest = WekaTools.loadLocalClassificationData("InsectSoundTEST.arff");
        System.out.println("Loaded Case Study data");
        //*/

        // Dataset Summaries
//        datasetSummaries(discreteDatasetNames, continuousDatasetNames, discreteCount, continuousCount,
//                         discreteData, continuousData, insectSoundTrain, insectSoundTest);
        

        // Ensembling VS Tree Tuning
        //ensembleVSTree(discreteDatasetNames, continuousDatasetNames, discreteCount, continuousCount,
//                       discreteData, continuousData);


        // CLASSIFIER COMPARISONS
//        compareClassifiers(discreteDatasetNames, continuousDatasetNames, discreteCount, continuousCount,
//                            discreteData, continuousData);


        // CASE STUDY
        caseStudy(insectSoundTrain, insectSoundTest);
    }
    
    private static void datasetSummaries(String[] discreteDatasetNames, String[] continuousDatasetNames,
                                         int discreteCount, int continuousCount,
                                         Instances[] discreteData, Instances[] continuousData,
                                         Instances insectSoundTrain, Instances insectSoundTest) {
        // Summary of datasets:
        /*
        System.out.println("Summary of Datasets");
        System.out.println("Discrete Datasets " + discreteCount);
        for (int discreteIt = 0; discreteIt < discreteCount; discreteIt++) {
//            System.out.println(discreteDatasetNames[discreteIt]);
//            System.out.println(discreteData[discreteIt].numAttributes()-1); // Attribute Count
//            System.out.println(discreteData[discreteIt].size()); // Instance Count
//            System.out.println(discreteData[discreteIt].numClasses()); // Classes
            // Distribution
            int[] classDistributionCount = new int[discreteData[discreteIt].numClasses()];
            for (Instance instance : discreteData[discreteIt]) {
                classDistributionCount[(int) instance.classValue()]++;
            }
            int instanceCount = discreteData[discreteIt].size();
            double[] classDistribution = new double[discreteData[discreteIt].numClasses()];
            String string = "";
            for (int i = 0; i < classDistribution.length; i++) {
                classDistribution[i] = (double) classDistributionCount[i] / instanceCount;
//                string += classDistributionCount[i] + ","; // Append count
                string += round(classDistribution[i] * 100) + ","; // Append proportions
            }
            string = string.substring(0, string.length() - 1);
            //System.out.println(string);
        }
        System.out.println("Continuous Datasets " + continuousCount);
        for (int continuousIt = 0; continuousIt < continuousCount; continuousIt++) {
//            System.out.println(continuousDatasetNames[continuousIt]-1);
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
        }
        //*/

        /* Specified Dataset:
        System.out.println("InsectSoundTrain Summary");
        SummariseData(insectSoundTrain);
        System.out.println("InsectSoundTest Summary");
        SummariseData(insectSoundTest);
        //*/

        // End of Summary
        //*/
        // PART 1)
        // Compare CourseworkTree, Weka ID3 and J48 classifiers
        /*
        System.out.println("Evaluation of Discrete Datasets");
        StringBuilder columns = new StringBuilder();
        StringBuilder[] accs = new StringBuilder[3];
        accs[0] = new StringBuilder();
        accs[1] = new StringBuilder();
        accs[2] = new StringBuilder();
        StringBuilder[] balAccs = new StringBuilder[3];
        balAccs[0] = new StringBuilder();
        balAccs[1] = new StringBuilder();
        balAccs[2] = new StringBuilder();

        for (int discreteIt = 0; discreteIt < discreteCount; discreteIt++) {
            Instances dataset = discreteData[discreteIt];
            Instances[] dataSplit = WekaTools.splitData(dataset, 0.8);

            //System.out.println("Evaluation of " + discreteDatasetNames[discreteIt] + ":");
            columns.append(discreteDatasetNames[discreteIt]).append("\t");

            CourseworkTree courseworkTree = new CourseworkTree();
            courseworkTree.setAttSplitMeasure(new ChiSquaredAttributeSplitMeasure());
            courseworkTree.buildClassifier(dataSplit[0]);
            ArrayList<int[]> ctConfMatrix = WekaTools.confusionMatrix(courseworkTree, dataSplit[1]);
            double ctAcc = WekaTools.accuracy(ctConfMatrix);
            double ctBalAcc = WekaTools.balancedAccuracy(ctConfMatrix);
            accs[0].append(ctAcc).append("\t");
            balAccs[0].append(ctBalAcc).append("\t");

            Id3 id3 = new Id3();
            id3.buildClassifier(dataSplit[0]);
            ArrayList<int[]> id3ConfMatrix = WekaTools.confusionMatrix(id3, dataSplit[1]);
            double id3Acc = WekaTools.accuracy(id3ConfMatrix);
            double id3BalAcc = WekaTools.balancedAccuracy(id3ConfMatrix);
            accs[1].append(id3Acc).append("\t");
            balAccs[1].append(id3BalAcc).append("\t");

            J48 j48 = new J48();
            j48.buildClassifier(dataSplit[0]);
            ArrayList<int[]> j48ConfMatrix = WekaTools.confusionMatrix(j48, dataSplit[1]);
            double j48Acc = WekaTools.accuracy(j48ConfMatrix);
            double j48BalAcc = WekaTools.balancedAccuracy(j48ConfMatrix);
            accs[2].append(j48Acc).append("\t");
            balAccs[2].append(j48BalAcc).append("\t");
        }
        System.out.println(columns);
        System.out.println(accs[0]);
        System.out.println(accs[1]);
        System.out.println(accs[2]);
        System.out.println(balAccs[0]);
        System.out.println(balAccs[1]);
        System.out.println(balAccs[2]);
        //*/

        // And for Continuous Datasets
        /*
        System.out.println("Evaluation of Continuous Datasets");
        StringBuilder columns = new StringBuilder();
        StringBuilder[] accs = new StringBuilder[3];
        accs[0] = new StringBuilder();
        accs[1] = new StringBuilder();
        accs[2] = new StringBuilder();
        StringBuilder[] balAccs = new StringBuilder[3];
        balAccs[0] = new StringBuilder();
        balAccs[1] = new StringBuilder();
        balAccs[2] = new StringBuilder();

        for (int continuousIt = 0; continuousIt < continuousCount; continuousIt++) {
            Instances dataset = continuousData[continuousIt];
            Instances[] dataSplit = WekaTools.splitData(dataset, 0.8);

            //System.out.println("Evaluation of " + continuousDatasetNames[continuousIt] + ":");
            columns.append(continuousDatasetNames[continuousIt]).append("\t");

            CourseworkTree courseworkTree = new CourseworkTree();
            courseworkTree.buildClassifier(dataSplit[0]);
            ArrayList<int[]> ctConfMatrix = WekaTools.confusionMatrix(courseworkTree, dataSplit[1]);
            double ctAcc = WekaTools.accuracy(ctConfMatrix);
            double ctBalAcc = WekaTools.balancedAccuracy(ctConfMatrix);
            accs[0].append(ctAcc).append("\t");
            balAccs[0].append(ctBalAcc).append("\t");

            SimpleCart simpleCart = new SimpleCart();
            simpleCart.buildClassifier(dataSplit[0]);
            ArrayList<int[]> simpCartConfMatrix = WekaTools.confusionMatrix(simpleCart, dataSplit[1]);
            double cartAcc = WekaTools.accuracy(simpCartConfMatrix);
            double cartBalAcc = WekaTools.balancedAccuracy(simpCartConfMatrix);
            accs[1].append(cartAcc).append("\t");
            balAccs[1].append(cartBalAcc).append("\t");

            J48 j48 = new J48();
            j48.buildClassifier(dataSplit[0]);
            ArrayList<int[]> j48ConfMatrix = WekaTools.confusionMatrix(j48, dataSplit[1]);
            double j48Acc = WekaTools.accuracy(j48ConfMatrix);
            double j48BalAcc = WekaTools.balancedAccuracy(j48ConfMatrix);
            accs[2].append(j48Acc).append("\t");
            balAccs[2].append(j48BalAcc).append("\t");
        }
        System.out.println(columns);
        System.out.println(accs[0]);
        System.out.println(accs[1]);
        System.out.println(accs[2]);
        System.out.println(balAccs[0]);
        System.out.println(balAccs[1]);
        System.out.println(balAccs[2]);
        //*/
    }

    private static void ensembleVSTree(String[] discreteDatasetNames, String[] continuousDatasetNames,
                                       int discreteCount, int continuousCount,
                                       Instances[] discreteData, Instances[] continuousData) throws Exception {
        ///*
        CourseworkTree ct = new CourseworkTree();
        TreeEnsemble treeEnsemble = new TreeEnsemble();

        StringBuilder columns = new StringBuilder("Accuracy");
        StringBuilder[] accs = new StringBuilder[8];
        accs[0] = new StringBuilder("50, 100%, false");
        accs[1] = new StringBuilder("50, 50%, false");
        accs[2] = new StringBuilder("25, 100%, false");
        accs[3] = new StringBuilder("100, 100%, false");
        accs[4] = new StringBuilder("IG");
        accs[5] = new StringBuilder("IGR");
        accs[6] = new StringBuilder("Chi");
        accs[7] = new StringBuilder("Gini");
        StringBuilder[] balAccs = new StringBuilder[8];
        balAccs[0] = new StringBuilder("50, 100%, false");
        balAccs[1] = new StringBuilder("50, 50%, false");
        balAccs[2] = new StringBuilder("25, 100%, false");
        balAccs[3] = new StringBuilder("100, 100%, false");
        balAccs[4] = new StringBuilder("IG");
        balAccs[5] = new StringBuilder("IGR");
        balAccs[6] = new StringBuilder("Chi");
        balAccs[7] = new StringBuilder("Gini");

        ArrayList<int[]> confMatrix;
        double acc, balAcc;

        ///* Discrete
        for (int discreteIt = 0; discreteIt < discreteCount; discreteIt++) {
            Instances[] trainTestSplit = WekaTools.splitData(discreteData[discreteIt], 0.8);

            columns.append("\t").append(discreteDatasetNames[discreteIt]);

            treeEnsemble.setNumTrees(50);
            treeEnsemble.setAttProp(1.0);
            treeEnsemble.setAverageDistributions(false);
            treeEnsemble.buildClassifier(trainTestSplit[0]);
            confMatrix = WekaTools.confusionMatrix(treeEnsemble, trainTestSplit[1]);
            acc = WekaTools.accuracy(confMatrix);
            balAcc = WekaTools.balancedAccuracy(confMatrix);
            accs[0].append("\t").append(acc);
            balAccs[0].append("\t").append(balAcc);

            treeEnsemble.setNumTrees(50);
            treeEnsemble.setAttProp(0.5);
            treeEnsemble.setAverageDistributions(false);
            treeEnsemble.buildClassifier(trainTestSplit[0]);
            confMatrix = WekaTools.confusionMatrix(treeEnsemble, trainTestSplit[1]);
            acc = WekaTools.accuracy(confMatrix);
            balAcc = WekaTools.balancedAccuracy(confMatrix);
            accs[1].append("\t").append(acc);
            balAccs[1].append("\t").append(balAcc);

            treeEnsemble.setNumTrees(25);
            treeEnsemble.setAttProp(1.0);
            treeEnsemble.setAverageDistributions(false);
            treeEnsemble.buildClassifier(trainTestSplit[0]);
            confMatrix = WekaTools.confusionMatrix(treeEnsemble, trainTestSplit[1]);
            acc = WekaTools.accuracy(confMatrix);
            balAcc = WekaTools.balancedAccuracy(confMatrix);
            accs[2].append("\t").append(acc);
            balAccs[2].append("\t").append(balAcc);

            treeEnsemble.setNumTrees(100);
            treeEnsemble.setAttProp(1.0);
            treeEnsemble.setAverageDistributions(false);
            treeEnsemble.buildClassifier(trainTestSplit[0]);
            confMatrix = WekaTools.confusionMatrix(treeEnsemble, trainTestSplit[1]);
            acc = WekaTools.accuracy(confMatrix);
            balAcc = WekaTools.balancedAccuracy(confMatrix);
            accs[3].append("\t").append(acc);
            balAccs[3].append("\t").append(balAcc);

            // Attribute Measure Split Criteria
//            ct.buildClassifier(trainTestSplit[0]);
//            confMatrix = WekaTools.confusionMatrix(ct, trainTestSplit[1]);
//            double ctAcc = WekaTools.accuracy(confMatrix);
//            double ctBalAcc = WekaTools.balancedAccuracy(confMatrix);
//            accs[4].append("\t").append(ctAcc);
//            balAccs[4].append("\t").append(ctBalAcc);
//
//            IGAttributeSplitMeasure igAttributeSplitMeasure = new IGAttributeSplitMeasure();
//            igAttributeSplitMeasure.setUseGain(false);
//            ct.setAttSplitMeasure(igAttributeSplitMeasure);
//            ct.buildClassifier(trainTestSplit[0]);
//            confMatrix = WekaTools.confusionMatrix(ct, trainTestSplit[1]);
//            ctAcc = WekaTools.accuracy(confMatrix);
//            ctBalAcc = WekaTools.balancedAccuracy(confMatrix);
//            accs[5].append("\t").append(ctAcc);
//            balAccs[5].append("\t").append(ctBalAcc);
//
//            ct.setAttSplitMeasure(new ChiSquaredAttributeSplitMeasure());
//            ct.buildClassifier(trainTestSplit[0]);
//            confMatrix = WekaTools.confusionMatrix(ct, trainTestSplit[1]);
//            ctAcc = WekaTools.accuracy(confMatrix);
//            ctBalAcc = WekaTools.balancedAccuracy(confMatrix);
//            accs[6].append("\t").append(ctAcc);
//            balAccs[6].append("\t").append(ctBalAcc);
//
//            ct.setAttSplitMeasure(igAttributeSplitMeasure);
//            ct.buildClassifier(trainTestSplit[0]);
//            confMatrix = WekaTools.confusionMatrix(ct, trainTestSplit[1]);
//            ctAcc = WekaTools.accuracy(confMatrix);
//            ctBalAcc = WekaTools.balancedAccuracy(confMatrix);
//            accs[7].append("\t").append(ctAcc);
//            balAccs[7].append("\t").append(ctBalAcc);
        }//*/
        /* Continuous Datasets
        for (int continuousIt = 0; continuousIt < continuousCount; continuousIt++) {
            Instances[] trainTestSplit = WekaTools.splitData(continuousData[continuousIt], 0.8);

            columns.append("\t").append(continuousDatasetNames[continuousIt]);

//            treeEnsemble.setNumTrees(50);
//            treeEnsemble.setAttProp(1.0);
//            treeEnsemble.setAverageDistributions(false);
//            treeEnsemble.buildClassifier(trainTestSplit[0]);
//            confMatrix = WekaTools.confusionMatrix(treeEnsemble, trainTestSplit[1]);
//            acc = WekaTools.accuracy(confMatrix);
//            balAcc = WekaTools.balancedAccuracy(confMatrix);
//            accs[0].append("\t").append(acc);
//            balAccs[0].append("\t").append(balAcc);
//
//            treeEnsemble.setNumTrees(50);
//            treeEnsemble.setAttProp(0.5);
//            treeEnsemble.setAverageDistributions(false);
//            treeEnsemble.buildClassifier(trainTestSplit[0]);
//            confMatrix = WekaTools.confusionMatrix(treeEnsemble, trainTestSplit[1]);
//            acc = WekaTools.accuracy(confMatrix);
//            balAcc = WekaTools.balancedAccuracy(confMatrix);
//            accs[1].append("\t").append(acc);
//            balAccs[1].append("\t").append(balAcc);

//            treeEnsemble.setNumTrees(25);
//            treeEnsemble.setAttProp(0.5);
//            treeEnsemble.setAverageDistributions(false);
//            treeEnsemble.buildClassifier(trainTestSplit[0]);
//            confMatrix = WekaTools.confusionMatrix(treeEnsemble, trainTestSplit[1]);
//            acc = WekaTools.accuracy(confMatrix);
//            balAcc = WekaTools.balancedAccuracy(confMatrix);
//            accs[2].append("\t").append(acc);
//            balAccs[2].append("\t").append(balAcc);
//
            // Note these attribute proportions are 0.5!!! But the title of it is not!
//            treeEnsemble.setNumTrees(100);
//            treeEnsemble.setAttProp(0.5);
//            treeEnsemble.setAverageDistributions(false);
//            treeEnsemble.buildClassifier(trainTestSplit[0]);
//            confMatrix = WekaTools.confusionMatrix(treeEnsemble, trainTestSplit[1]);
//            acc = WekaTools.accuracy(confMatrix);
//            balAcc = WekaTools.balancedAccuracy(confMatrix);
//            accs[3].append("\t").append(acc);
//            balAccs[3].append("\t").append(balAcc);


            // extra one (do not run with other code in this segment)
//            treeEnsemble.setNumTrees(50);
//            treeEnsemble.setAttProp(0.25);
//            treeEnsemble.setAverageDistributions(false);
//            treeEnsemble.buildClassifier(trainTestSplit[0]);
//            confMatrix = WekaTools.confusionMatrix(treeEnsemble, trainTestSplit[1]);
//            acc = WekaTools.accuracy(confMatrix);
//            balAcc = WekaTools.balancedAccuracy(confMatrix);
//            accs[3].append("\t").append(acc);
//            balAccs[3].append("\t").append(balAcc);


            // Attribute Measure Split Criteria
//            ct.buildClassifier(trainTestSplit[0]);
//            confMatrix = WekaTools.confusionMatrix(ct, trainTestSplit[1]);
//            double ctAcc = WekaTools.accuracy(confMatrix);
//            double ctBalAcc = WekaTools.balancedAccuracy(confMatrix);
//            accs[4].append("\t").append(ctAcc);
//            balAccs[4].append("\t").append(ctBalAcc);
//
//            IGAttributeSplitMeasure igAttributeSplitMeasure = new IGAttributeSplitMeasure();
//            igAttributeSplitMeasure.setUseGain(false);
//            ct.setAttSplitMeasure(igAttributeSplitMeasure);
//            ct.buildClassifier(trainTestSplit[0]);
//            confMatrix = WekaTools.confusionMatrix(ct, trainTestSplit[1]);
//            ctAcc = WekaTools.accuracy(confMatrix);
//            ctBalAcc = WekaTools.balancedAccuracy(confMatrix);
//            accs[5].append("\t").append(ctAcc);
//            balAccs[5].append("\t").append(ctBalAcc);
//
//            ct.setAttSplitMeasure(new ChiSquaredAttributeSplitMeasure());
//            ct.buildClassifier(trainTestSplit[0]);
//            confMatrix = WekaTools.confusionMatrix(ct, trainTestSplit[1]);
//            ctAcc = WekaTools.accuracy(confMatrix);
//            ctBalAcc = WekaTools.balancedAccuracy(confMatrix);
//            accs[6].append("\t").append(ctAcc);
//            balAccs[6].append("\t").append(ctBalAcc);
//
//            ct.setAttSplitMeasure(igAttributeSplitMeasure);
//            ct.buildClassifier(trainTestSplit[0]);
//            confMatrix = WekaTools.confusionMatrix(ct, trainTestSplit[1]);
//            ctAcc = WekaTools.accuracy(confMatrix);
//            ctBalAcc = WekaTools.balancedAccuracy(confMatrix);
//            accs[7].append("\t").append(ctAcc);
//            balAccs[7].append("\t").append(ctBalAcc);
        }//*/
        System.out.println(columns);
        for (StringBuilder accString : accs) {
            System.out.println(accString);
        }
        for (StringBuilder balAccString : balAccs) {
            System.out.println(balAccString);
        }
    }

    private static void compareClassifiers(String[] discreteDatasetNames, String[] continuousDatasetNames,
                                           int discreteCount, int continuousCount,
                                           Instances[] discreteData, Instances[] continuousData) throws Exception {
        // Contains times for each classifier on each problem
        // This was to check for classifications that took too long and ignore them or do them later
        ///*
        StringBuilder columns = new StringBuilder("Measure");

        StringBuilder[] accs = new StringBuilder[17];
        accs[0] = new StringBuilder("TreeEnsemble");
        accs[1] = new StringBuilder("ID3/CART");
        accs[2] = new StringBuilder("J48");
        accs[3] = new StringBuilder("IBk");
        accs[4] = new StringBuilder("PART");
        accs[5] = new StringBuilder("NaiveBayes");
        accs[6] = new StringBuilder("OneR");
        accs[7] = new StringBuilder("SMO");
        accs[8] = new StringBuilder("Logistic");
        accs[9] = new StringBuilder("LogitBoost");
        accs[10] = new StringBuilder("DecisionStump");
        accs[11] = new StringBuilder("Bagging");
        accs[12] = new StringBuilder("RandomForest");
        accs[13] = new StringBuilder("RotationForest");
        accs[14] = new StringBuilder("AdaBoostM1");
        accs[15] = new StringBuilder("Voting");
        accs[16] = new StringBuilder("Stacking");

        StringBuilder[] balAccs = new StringBuilder[17];
        balAccs[0] = new StringBuilder("TreeEnsemble");
        balAccs[1] = new StringBuilder("IB3/CART");
        balAccs[2] = new StringBuilder("J48");
        balAccs[3] = new StringBuilder("IBk");
        balAccs[4] = new StringBuilder("PART");
        balAccs[5] = new StringBuilder("NaiveBayes");
        balAccs[6] = new StringBuilder("OneR");
        balAccs[7] = new StringBuilder("SMO");
        balAccs[8] = new StringBuilder("Logistic");
        balAccs[9] = new StringBuilder("LogitBoost");
        balAccs[10] = new StringBuilder("DecisionStump");
        balAccs[11] = new StringBuilder("Bagging");
        balAccs[12] = new StringBuilder("RandomForest");
        balAccs[13] = new StringBuilder("RotationForest");
        balAccs[14] = new StringBuilder("AdaBoostM1");
        balAccs[15] = new StringBuilder("Voting");
        balAccs[16] = new StringBuilder("Stacking");

        long startTime;
        /* Discrete Datasets
        for (int discreteIt = 0; discreteIt < discreteCount; discreteIt++) { // Modify the iterator for SMO, Logistic and RotForest
            // these takes a while for some of the problems
            Instances[] trainTestSplit = WekaTools.splitData(discreteData[discreteIt], 0.8);

            columns.append("\t").append(discreteDatasetNames[discreteIt]);
            System.out.println(discreteDatasetNames[discreteIt] + "(" + discreteIt + ")");

            startTime = System.currentTimeMillis();
            TreeEnsemble treeEnsemble = new TreeEnsemble();
            treeEnsemble.setAttributeSplitMeasure("chi");
            treeEnsemble.setNumTrees(25);
            treeEnsemble.setAttProp(1.0);
            treeEnsemble.buildClassifier(trainTestSplit[0]);
            ArrayList<int[]> confMatrix = WekaTools.confusionMatrix(treeEnsemble, trainTestSplit[1]);
//            for (int[] classLine : confMatrix) { // Debug: confusion matrix
//                System.out.println(Arrays.toString(classLine));
//            }
            double acc = WekaTools.accuracy(confMatrix);
            double balAcc = WekaTools.balancedAccuracy(confMatrix);
            accs[0].append("\t").append(acc);
            balAccs[0].append("\t").append(balAcc);
            System.out.print("TreeEnsemble(" + (System.currentTimeMillis()-startTime) + "),");

//            startTime = System.currentTimeMillis();
//            Id3 id3 = new Id3();
//            id3.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> id3ConfMatrix = WekaTools.confusionMatrix(id3, trainTestSplit[1]);
//            double id3Acc = WekaTools.accuracy(id3ConfMatrix);
//            double id3BalAcc = WekaTools.balancedAccuracy(id3ConfMatrix);
//            accs[1].append("\t").append(id3Acc);
//            balAccs[1].append("\t").append(id3BalAcc);
//            System.out.print("ID3(" + (System.currentTimeMillis()-startTime) + "),");
//
//
//            startTime = System.currentTimeMillis();
//            IBk ibk = new IBk(); // k-NN
//            ibk.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> ibkConfMatrix = WekaTools.confusionMatrix(ibk, trainTestSplit[1]);
////            for (int[] classLine : ibkConfMatrix) { // Debug: confusion matrix
////                System.out.println(Arrays.toString(classLine));
////            }
//            double ibkAcc = WekaTools.accuracy(ibkConfMatrix);
//            double ibkBalAcc = WekaTools.balancedAccuracy(ibkConfMatrix);
//            accs[3].append("\t").append(ibkAcc);
//            balAccs[3].append("\t").append(ibkBalAcc);
//            System.out.print("IBk(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            J48 j48 = new J48(); // C4.5 decision tree(s)
//            j48.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> j48ConfMatrix = WekaTools.confusionMatrix(j48, trainTestSplit[1]);
//            double j48Acc = WekaTools.accuracy(j48ConfMatrix);
//            double j48BalAcc = WekaTools.balancedAccuracy(j48ConfMatrix);
//            accs[2].append("\t").append(j48Acc);
//            balAccs[2].append("\t").append(j48BalAcc);
//            System.out.print("J48(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            PART part = new PART(); // rule learner???
//            part.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> partConfMatrix = WekaTools.confusionMatrix(part, trainTestSplit[1]);
//            double partAcc = WekaTools.accuracy(partConfMatrix);
//            double partBalAcc = WekaTools.balancedAccuracy(partConfMatrix);
//            accs[4].append("\t").append(partAcc);
//            balAccs[4].append("\t").append(partBalAcc);
//            System.out.print("PART(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            NaiveBayes naiveBayes = new NaiveBayes(); // With/without kernels
//            naiveBayes.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> naiveConfMatrix = WekaTools.confusionMatrix(naiveBayes, trainTestSplit[1]);
//            double naiveAcc = WekaTools.accuracy(naiveConfMatrix);
//            double naiveBalAcc = WekaTools.balancedAccuracy(naiveConfMatrix);
//            accs[5].append("\t").append(naiveAcc);
//            balAccs[5].append("\t").append(naiveBalAcc);
//            System.out.print("Naive(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            OneR oneR = new OneR(); // Holte's OneR
//            oneR.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> oneConfMatrix = WekaTools.confusionMatrix(oneR, trainTestSplit[1]);
//            double oneRAcc = WekaTools.accuracy(oneConfMatrix);
//            double oneBalAcc = WekaTools.balancedAccuracy(oneConfMatrix);
//            accs[6].append("\t").append(oneRAcc);
//            balAccs[7].append("\t").append(oneBalAcc);
//            System.out.print("OneR(" + (System.currentTimeMillis()-startTime) + "),");

            // SMO, Logistic long time classifiers on some datasets
//            if (!(discreteIt == 1 || discreteIt == 3 || discreteIt == 17)) {
//                accs[7].append("\t").append("N/A");
//                balAccs[7].append("\t").append("N/A");
//                System.out.println("SMO SKIPPED");
//            } else {
//                startTime = System.currentTimeMillis();
//                SMO smo = new SMO(); // SVM
//                smo.buildClassifier(trainTestSplit[0]);
//                ArrayList<int[]> smoConfMatrix = WekaTools.confusionMatrix(smo, trainTestSplit[1]);
//                double smoAcc = WekaTools.accuracy(smoConfMatrix);
//                double smoBalAcc = WekaTools.balancedAccuracy(smoConfMatrix);
//                accs[7].append("\t").append(smoAcc);
//                balAccs[7].append("\t").append(smoBalAcc);
//                System.out.println("SMO(" + (System.currentTimeMillis() - startTime) + "): " + +smoAcc + ", " + smoBalAcc);
//            }
//
//            if (!(discreteIt == 1 || discreteIt == 3 || discreteIt == 16 || discreteIt == 17 || discreteIt == 18)) {
//                accs[8].append("\t").append("N/A");
//                balAccs[8].append("\t").append("N/A");
//                System.out.println("Logistic SKIPPED");
//            } else {
//                startTime = System.currentTimeMillis();
//                Logistic logistic = new Logistic(); // Logistic Regression
//                logistic.buildClassifier(trainTestSplit[0]);
//                ArrayList<int[]> logisticConfMatrix = WekaTools.confusionMatrix(logistic, trainTestSplit[1]);
//                double logisticAcc = WekaTools.accuracy(logisticConfMatrix);
//                double logisticBalAcc = WekaTools.balancedAccuracy(logisticConfMatrix);
//                accs[8].append("\t").append(logisticAcc);
//                balAccs[8].append("\t").append(logisticBalAcc);
//                System.out.println("Logistic(" + (System.currentTimeMillis() - startTime) + "): " + +logisticAcc + ", " + logisticBalAcc);
//            }

//            startTime = System.currentTimeMillis();
//            LogitBoost logitBoost = new LogitBoost();
//            logitBoost.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> logitConfMatrix = WekaTools.confusionMatrix(logitBoost, trainTestSplit[1]);
//            double logitAcc = WekaTools.accuracy(logitConfMatrix);
//            double logitBalAcc = WekaTools.balancedAccuracy(logitConfMatrix);
//            accs[9].append("\t").append(logitAcc);
//            balAccs[9].append("\t").append(logitBalAcc);
//            System.out.print("Logit(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            DecisionStump decisionStump = new DecisionStump(); // For boosting?
//            decisionStump.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> stumpConfMatrix = WekaTools.confusionMatrix(decisionStump, trainTestSplit[1]);
//            double stumpAcc = WekaTools.accuracy(stumpConfMatrix);
//            double stumpBalAcc = WekaTools.balancedAccuracy(stumpConfMatrix);
//            accs[10].append("\t").append(stumpAcc);
//            balAccs[10].append("\t").append(stumpBalAcc);
//            System.out.print("Stump(" + (System.currentTimeMillis()-startTime) + "),");
//
//            // Ensemble Classifiers
//            startTime = System.currentTimeMillis();
//            Bagging bagging = new Bagging();
//            bagging.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> baggingConfMatrix = WekaTools.confusionMatrix(bagging, trainTestSplit[1]);
//            double baggingAcc = WekaTools.accuracy(bagging, trainTestSplit[1]);
//            double baggingBalAcc = WekaTools.balancedAccuracy(baggingConfMatrix);
//            accs[11].append("\t").append(baggingAcc);
//            balAccs[11].append("\t").append(baggingBalAcc);
//            System.out.print("Bag(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            RandomForest randomForest = new RandomForest();
//            randomForest.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> ranForConfMatrix = WekaTools.confusionMatrix(randomForest, trainTestSplit[1]);
//            double ranForAcc = WekaTools.accuracy(ranForConfMatrix);
//            double ranForBalAcc = WekaTools.balancedAccuracy(ranForConfMatrix);
//            accs[12].append("\t").append(ranForAcc);
//            balAccs[12].append("\t").append(ranForBalAcc);
//            System.out.print("RanFor(" + (System.currentTimeMillis()-startTime) + "),");

            // Rotation forest takes some time on some of the datasets
//            if (!(discreteIt == 1 || discreteIt == 3 || discreteIt == 16 || discreteIt == 17)) {
//                accs[13].append("\t").append("N/A");
//                balAccs[13].append("\t").append("N/A");
//                System.out.println("RotForest SKIPPED");
//            } else {
//                startTime = System.currentTimeMillis();
//                RotationForest rotationForest = new RotationForest();
//                rotationForest.buildClassifier(trainTestSplit[0]);
//                ArrayList<int[]> rotForConfMatrix = WekaTools.confusionMatrix(rotationForest, trainTestSplit[1]);
//                double rotForAcc = WekaTools.accuracy(rotForConfMatrix);
//                double rotForBalAcc = WekaTools.balancedAccuracy(rotForConfMatrix);
//                System.out.println("Rotation Forest(" + (System.currentTimeMillis() - startTime) + "): " + rotForAcc + ", " + rotForBalAcc);
//                accs[13].append("\t").append(rotForAcc);
//                balAccs[13].append("\t").append(rotForBalAcc);
//            }

//            startTime = System.currentTimeMillis();
//            AdaBoostM1 adaBoostM1 = new AdaBoostM1();
//            adaBoostM1.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> adaConfMatrix = WekaTools.confusionMatrix(adaBoostM1, trainTestSplit[1]);
//            double adaAcc = WekaTools.accuracy(adaConfMatrix);
//            double adaBalAcc = WekaTools.balancedAccuracy(adaConfMatrix);
//            accs[14].append("\t").append(adaAcc);
//            balAccs[14].append("\t").append(adaBalAcc);
//            System.out.print("Ada(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            Vote vote = new Vote();
//            vote.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> voteConfMatrix = WekaTools.confusionMatrix(vote, trainTestSplit[1]);
//            double voteAcc = WekaTools.accuracy(voteConfMatrix);
//            double voteBalAcc = WekaTools.balancedAccuracy(voteConfMatrix);
//            accs[15].append("\t").append(voteAcc);
//            balAccs[15].append("\t").append(voteBalAcc);
//            System.out.print("Vote(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            Stacking stacking = new Stacking();
//            stacking.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> stackConfMatrix = WekaTools.confusionMatrix(stacking, trainTestSplit[1]);
//            double stackAcc = WekaTools.accuracy(stackConfMatrix);
//            double stackBalAcc = WekaTools.balancedAccuracy(stackConfMatrix);
//            accs[16].append("\t").append(stackAcc);
//            balAccs[16].append("\t").append(stackBalAcc);
//            System.out.print("Stack(" + (System.currentTimeMillis()-startTime) + "),");

            System.out.println();
        }
        System.out.println(columns);
        for (StringBuilder accString : accs) {
            System.out.println(accString);
        }
        for (StringBuilder balAccString : balAccs) {
            System.out.println(balAccString);
        }
        //*/

        ///* Continuous datasets
        for (int continuousIt = 0; continuousIt < continuousCount; continuousIt++) {
            Instances[] trainTestSplit = WekaTools.splitData(continuousData[continuousIt], 0.8);

            columns.append("\t").append(continuousDatasetNames[continuousIt]);
            System.out.println(continuousDatasetNames[continuousIt]);

            startTime = System.currentTimeMillis();
            TreeEnsemble treeEnsemble = new TreeEnsemble();
            treeEnsemble.setAttributeSplitMeasure("chi");
            treeEnsemble.setNumTrees(25);
            treeEnsemble.buildClassifier(trainTestSplit[0]);
            ArrayList<int[]> confMatrix = WekaTools.confusionMatrix(treeEnsemble, trainTestSplit[1]);
            double acc = WekaTools.accuracy(confMatrix);
            double balAcc = WekaTools.balancedAccuracy(confMatrix);
            accs[0].append("\t").append(acc);
            balAccs[0].append("\t").append(balAcc);
            System.out.print("TreeEnsemble(" + (System.currentTimeMillis()-startTime) + "),");

//            startTime = System.currentTimeMillis();
//            SimpleCart sCART = new SimpleCart();
//            sCART.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> sCARTConfMatrix = WekaTools.confusionMatrix(sCART, trainTestSplit[1]);
//            double sCARTAcc = WekaTools.accuracy(sCARTConfMatrix);
//            double sCARTBalAcc = WekaTools.balancedAccuracy(sCARTConfMatrix);
//            accs[1].append("\t").append(sCARTAcc);
//            balAccs[1].append("\t").append(sCARTBalAcc);
//            System.out.print("SimpleCART(" + (System.currentTimeMillis()-startTime) + "),");

//            startTime = System.currentTimeMillis();
//            J48 j48 = new J48(); // C4.5 decision tree(s)
//            j48.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> j48ConfMatrix = WekaTools.confusionMatrix(j48, trainTestSplit[1]);
//            double j48Acc = WekaTools.accuracy(j48ConfMatrix);
//            double j48BalAcc = WekaTools.balancedAccuracy(j48ConfMatrix);
//            accs[2].append("\t").append(j48Acc);
//            balAccs[2].append("\t").append(j48BalAcc);
//            System.out.print("J48(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            IBk ibk = new IBk(); // k-NN
//            ibk.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> ibkConfMatrix = WekaTools.confusionMatrix(ibk, trainTestSplit[1]);
//            double ibkAcc = WekaTools.accuracy(ibkConfMatrix);
//            double ibkBalAcc = WekaTools.balancedAccuracy(ibkConfMatrix);
//            accs[3].append("\t").append(ibkAcc);
//            balAccs[3].append("\t").append(ibkBalAcc);
//            System.out.print("IBk(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            PART part = new PART(); // rule learner???
//            part.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> partConfMatrix = WekaTools.confusionMatrix(part, trainTestSplit[1]);
//            double partAcc = WekaTools.accuracy(partConfMatrix);
//            double partBalAcc = WekaTools.balancedAccuracy(partConfMatrix);
//            accs[4].append("\t").append(partAcc);
//            balAccs[4].append("\t").append(partBalAcc);
//            System.out.print("PART(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            NaiveBayes naiveBayes = new NaiveBayes(); // With/without kernels
//            naiveBayes.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> naiveConfMatrix = WekaTools.confusionMatrix(naiveBayes, trainTestSplit[1]);
//            double naiveAcc = WekaTools.accuracy(naiveConfMatrix);
//            double naiveBalAcc = WekaTools.balancedAccuracy(naiveConfMatrix);
//            accs[5].append("\t").append(naiveAcc);
//            balAccs[5].append("\t").append(naiveBalAcc);
//            System.out.print("Naive(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            OneR oneR = new OneR(); // Holte's OneR
//            oneR.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> oneConfMatrix = WekaTools.confusionMatrix(oneR, trainTestSplit[1]);
//            double oneRAcc = WekaTools.accuracy(oneConfMatrix);
//            double oneBalAcc = WekaTools.balancedAccuracy(oneConfMatrix);
//            accs[6]].append("\t").append(oneRAcc);
//            balAccs[6].append("\t").append(oneBalAcc);
//            System.out.print("OneR(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            SMO smo = new SMO(); // SVM
//            smo.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> smoConfMatrix = WekaTools.confusionMatrix(smo, trainTestSplit[1]);
//            double smoAcc = WekaTools.accuracy(smoConfMatrix);
//            double smoBalAcc = WekaTools.balancedAccuracy(smoConfMatrix);
//            accs[7].append("\t").append(smoAcc);
//            balAccs[7].append("\t").append(smoBalAcc);
//            System.out.print("SMO(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            Logistic logistic = new Logistic(); // Logistic Regression
//            logistic.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> logisticConfMatrix = WekaTools.confusionMatrix(logistic, trainTestSplit[1]);
//            double logisticAcc = WekaTools.accuracy(logisticConfMatrix);
//            double logisticBalAcc = WekaTools.balancedAccuracy(logisticConfMatrix);
//            accs[8].append("\t").append(logisticAcc);
//            balAccs[8].append("\t").append(logisticBalAcc);
//            System.out.print("Logistic(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            LogitBoost logitBoost = new LogitBoost();
//            logitBoost.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> logitConfMatrix = WekaTools.confusionMatrix(logitBoost, trainTestSplit[1]);
//            double logitAcc = WekaTools.accuracy(logitConfMatrix);
//            double logitBalAcc = WekaTools.balancedAccuracy(logitConfMatrix);
//            accs[9].append("\t").append(logitAcc);
//            balAccs[9].append("\t").append(logitBalAcc);
//            System.out.print("Logit(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            DecisionStump decisionStump = new DecisionStump(); // For boosting?
//            decisionStump.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> stumpConfMatrix = WekaTools.confusionMatrix(decisionStump, trainTestSplit[1]);
//            double stumpAcc = WekaTools.accuracy(stumpConfMatrix);
//            double stumpBalAcc = WekaTools.balancedAccuracy(stumpConfMatrix);
//            accs[10].append("\t").append(stumpAcc);
//            balAccs[10].append("\t").append(stumpBalAcc);
//            System.out.print("Stump(" + (System.currentTimeMillis()-startTime) + "),");
//
//            // Ensemble Classifiers
//            startTime = System.currentTimeMillis();
//            Bagging bagging = new Bagging();
//            bagging.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> baggingConfMatrix = WekaTools.confusionMatrix(bagging, trainTestSplit[1]);
//            double baggingAcc = WekaTools.accuracy(bagging, trainTestSplit[1]);
//            double baggingBalAcc = WekaTools.balancedAccuracy(baggingConfMatrix);
//            accs[11].append("\t").append(baggingAcc);
//            balAccs[11].append("\t").append(baggingBalAcc);
//            System.out.print("Bag(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            RandomForest randomForest = new RandomForest();
//            randomForest.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> ranForConfMatrix = WekaTools.confusionMatrix(randomForest, trainTestSplit[1]);
//            double ranForAcc = WekaTools.accuracy(ranForConfMatrix);
//            double ranForBalAcc = WekaTools.balancedAccuracy(ranForConfMatrix);
//            accs[12].append("\t").append(ranForAcc);
//            balAccs[12].append("\t").append(ranForBalAcc);
//            System.out.print("RanFor(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            RotationForest rotationForest = new RotationForest();
//            rotationForest.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> rotForConfMatrix = WekaTools.confusionMatrix(rotationForest, trainTestSplit[1]);
//            double rotForAcc = WekaTools.accuracy(rotForConfMatrix);
//            double rotForBalAcc = WekaTools.balancedAccuracy(rotForConfMatrix);
//            accs[13].append("\t").append(rotForAcc);
//            balAccs[13].append("\t").append(rotForBalAcc);
//            System.out.print("RotFor(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            AdaBoostM1 adaBoostM1 = new AdaBoostM1();
//            adaBoostM1.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> adaConfMatrix = WekaTools.confusionMatrix(adaBoostM1, trainTestSplit[1]);
//            double adaAcc = WekaTools.accuracy(adaConfMatrix);
//            double adaBalAcc = WekaTools.balancedAccuracy(adaConfMatrix);
//            accs[14].append("\t").append(adaAcc);
//            balAccs[14].append("\t").append(adaBalAcc);
//            System.out.print("Ada(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            Vote vote = new Vote();
//            vote.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> voteConfMatrix = WekaTools.confusionMatrix(vote, trainTestSplit[1]);
//            double voteAcc = WekaTools.accuracy(voteConfMatrix);
//            double voteBalAcc = WekaTools.balancedAccuracy(voteConfMatrix);
//            accs[15].append("\t").append(voteAcc);
//            balAccs[15].append("\t").append(voteBalAcc);
//            System.out.print("Vote(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            Stacking stacking = new Stacking();
//            stacking.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> stackConfMatrix = WekaTools.confusionMatrix(stacking, trainTestSplit[1]);
//            double stackAcc = WekaTools.accuracy(stackConfMatrix);
//            double stackBalAcc = WekaTools.balancedAccuracy(stackConfMatrix);
//            accs[16].append("\t").append(stackAcc);
//            balAccs[16].append("\t").append(stackBalAcc);
//            System.out.print("Stack(" + (System.currentTimeMillis()-startTime) + "),");

            System.out.println();
        }
        System.out.println(columns);
        for (StringBuilder accString : accs) {
            System.out.println(accString);
        }
        for (StringBuilder balAccString : balAccs) {
            System.out.println(balAccString);
        }
        //*/
    }

    private static void caseStudy(Instances insectSoundTrain, Instances insectSoundTest) throws Exception {
        /* Find best attribute measure to split on
        CourseworkTree courseworkTree = new CourseworkTree();
        IGAttributeSplitMeasure igAttributeSplitMeasure = new IGAttributeSplitMeasure();
        courseworkTree.setAttSplitMeasure(igAttributeSplitMeasure);
        courseworkTree.buildClassifier(insectSoundTrain);
        ArrayList<int[]> confMatrix = WekaTools.confusionMatrix(courseworkTree, insectSoundTest);
        double acc = WekaTools.accuracy(confMatrix);
        double balAcc = WekaTools.balancedAccuracy(confMatrix);
        System.out.println("IG\t" + acc + "\t" + balAcc);

        igAttributeSplitMeasure.setUseGain(false);
        courseworkTree.setAttSplitMeasure(igAttributeSplitMeasure);
        courseworkTree.buildClassifier(insectSoundTrain);
        confMatrix = WekaTools.confusionMatrix(courseworkTree, insectSoundTest);
        acc = WekaTools.accuracy(confMatrix);
        balAcc = WekaTools.balancedAccuracy(confMatrix);
        System.out.println("IGR\t" + acc + "\t" + balAcc);

        ChiSquaredAttributeSplitMeasure chiSquaredAttributeSplitMeasure = new ChiSquaredAttributeSplitMeasure();
        courseworkTree.setAttSplitMeasure(chiSquaredAttributeSplitMeasure);
        courseworkTree.buildClassifier(insectSoundTrain);
        confMatrix = WekaTools.confusionMatrix(courseworkTree, insectSoundTest);
        acc = WekaTools.accuracy(confMatrix);
        balAcc = WekaTools.balancedAccuracy(confMatrix);
        System.out.println("Chi\t" + acc + "\t" + balAcc);

        GiniAttributeSplitMeasure giniAttributeSplitMeasure = new GiniAttributeSplitMeasure();
        courseworkTree.setAttSplitMeasure(giniAttributeSplitMeasure);
        courseworkTree.buildClassifier(insectSoundTrain);
        confMatrix = WekaTools.confusionMatrix(courseworkTree, insectSoundTest);
        acc = WekaTools.accuracy(confMatrix);
        balAcc = WekaTools.balancedAccuracy(confMatrix);
        System.out.println("Gini\t" + acc + "\t" + balAcc);
        //*/

        ///* Against other Classifiers
        StringBuilder[] accs = new StringBuilder[16];
        accs[0] = new StringBuilder("CourseworkTree");
        accs[1] = new StringBuilder("TreeEnsemble");
        accs[2] = new StringBuilder("CART");
        accs[3] = new StringBuilder("J48");
        accs[4] = new StringBuilder("IBk");
        accs[5] = new StringBuilder("PART");
        accs[6] = new StringBuilder("NaiveBayes");
        accs[7] = new StringBuilder("OneR");
        accs[8] = new StringBuilder("SMO");
        accs[9] = new StringBuilder("Logistic");
        accs[10] = new StringBuilder("LogitBoost");
        accs[11] = new StringBuilder("DecisionStump");
        accs[12] = new StringBuilder("Bagging");
        accs[13] = new StringBuilder("RandomForest");
        accs[14] = new StringBuilder("RotationForest");
        accs[15] = new StringBuilder("AdaBoostM1");

        StringBuilder[] balAccs = new StringBuilder[16];
        balAccs[0] = new StringBuilder("CourseworkTree");
        balAccs[1] = new StringBuilder("TreeEnsemble");
        balAccs[2] = new StringBuilder("SimpleCart");
        balAccs[3] = new StringBuilder("J48");
        balAccs[4] = new StringBuilder("IBk");
        balAccs[5] = new StringBuilder("PART");
        balAccs[6] = new StringBuilder("NaiveBayes");
        balAccs[7] = new StringBuilder("OneR");
        balAccs[8] = new StringBuilder("SMO");
        balAccs[9] = new StringBuilder("Logistic");
        balAccs[10] = new StringBuilder("LogitBoost");
        balAccs[11] = new StringBuilder("DecisionStump");
        balAccs[12] = new StringBuilder("Bagging");
        balAccs[13] = new StringBuilder("RandomForest");
        balAccs[14] = new StringBuilder("RotationForest");
        balAccs[15] = new StringBuilder("AdaBoostM1");
        
        StringBuilder[] nlls = new StringBuilder[16];
        nlls[0] = new StringBuilder("CourseworkTree");
        nlls[1] = new StringBuilder("TreeEnsemble");
        nlls[2] = new StringBuilder("SimpleCart");
        nlls[3] = new StringBuilder("J48");
        nlls[4] = new StringBuilder("IBk");
        nlls[5] = new StringBuilder("PART");
        nlls[6] = new StringBuilder("NaiveBayes");
        nlls[7] = new StringBuilder("OneR");
        nlls[8] = new StringBuilder("SMO");
        nlls[9] = new StringBuilder("Logistic");
        nlls[10] = new StringBuilder("LogitBoost");
        nlls[11] = new StringBuilder("DecisionStump");
        nlls[12] = new StringBuilder("Bagging");
        nlls[13] = new StringBuilder("RandomForest");
        nlls[14] = new StringBuilder("RotationForest");
        nlls[15] = new StringBuilder("AdaBoostM1");

        long startTime;

        startTime = System.currentTimeMillis();
        CourseworkTree courseworkTree = new CourseworkTree();
        GiniAttributeSplitMeasure giniAttributeSplitMeasure = new GiniAttributeSplitMeasure();
        courseworkTree.setAttSplitMeasure(giniAttributeSplitMeasure);
        courseworkTree.buildClassifier(insectSoundTrain);
        ArrayList<int[]> ctConfMatrix = WekaTools.confusionMatrix(courseworkTree, insectSoundTest);
        double ctCcc = WekaTools.accuracy(ctConfMatrix);
        double ctBalAcc = WekaTools.balancedAccuracy(ctConfMatrix);
        accs[0].append("\t").append(ctCcc);
        balAccs[0].append("\t").append(ctBalAcc);
        nlls[0].append("\t").append(WekaTools.nll(courseworkTree, insectSoundTest));
        System.out.print("CourseworkTree(" + (System.currentTimeMillis()-startTime) + "),");

        startTime = System.currentTimeMillis();
        TreeEnsemble treeEnsemble = new TreeEnsemble();
        treeEnsemble.setAttributeSplitMeasure("gini");
        treeEnsemble.setNumTrees(10);
        treeEnsemble.buildClassifier(insectSoundTrain);
        ArrayList<int[]> teConfMatrix = WekaTools.confusionMatrix(treeEnsemble, insectSoundTest);
        double teAcc = WekaTools.accuracy(teConfMatrix);
        double teBalAcc = WekaTools.balancedAccuracy(teConfMatrix);
        accs[1].append("\t").append(teAcc);
        balAccs[1].append("\t").append(teBalAcc);
        nlls[1].append("\t").append(WekaTools.nll(treeEnsemble, insectSoundTest));
        System.out.print("TreeEnsemble(" + (System.currentTimeMillis()-startTime) + "),");

        startTime = System.currentTimeMillis();
        SimpleCart sCART = new SimpleCart();
        sCART.buildClassifier(insectSoundTrain);
        ArrayList<int[]> sCARTConfMatrix = WekaTools.confusionMatrix(sCART, insectSoundTest);
        double sCARTAcc = WekaTools.accuracy(sCARTConfMatrix);
        double sCARTBalAcc = WekaTools.balancedAccuracy(sCARTConfMatrix);
        accs[2].append("\t").append(sCARTAcc);
        balAccs[2].append("\t").append(sCARTBalAcc);
        nlls[2].append("\t").append(WekaTools.nll(sCART, insectSoundTest));
        System.out.print("SimpleCART(" + (System.currentTimeMillis()-startTime) + "),");

        startTime = System.currentTimeMillis();
        J48 j48 = new J48(); // C4.5 decision tree(s)
        j48.buildClassifier(insectSoundTrain);
        ArrayList<int[]> j48ConfMatrix = WekaTools.confusionMatrix(j48, insectSoundTest);
        double j48Acc = WekaTools.accuracy(j48ConfMatrix);
        double j48BalAcc = WekaTools.balancedAccuracy(j48ConfMatrix);
        accs[3].append("\t").append(j48Acc);
        balAccs[3].append("\t").append(j48BalAcc);
        nlls[3].append("\t").append(WekaTools.nll(j48, insectSoundTest));
        System.out.print("J48(" + (System.currentTimeMillis()-startTime) + "),");

        startTime = System.currentTimeMillis();
        IBk ibk = new IBk(); // k-NN
        ibk.buildClassifier(insectSoundTrain);
        ArrayList<int[]> ibkConfMatrix = WekaTools.confusionMatrix(ibk, insectSoundTest);
        double ibkAcc = WekaTools.accuracy(ibkConfMatrix);
        double ibkBalAcc = WekaTools.balancedAccuracy(ibkConfMatrix);
        accs[4].append("\t").append(ibkAcc);
        balAccs[4].append("\t").append(ibkBalAcc);
        nlls[4].append("\t").append(WekaTools.nll(ibk, insectSoundTest));
        System.out.print("IBk(" + (System.currentTimeMillis()-startTime) + "),");

        startTime = System.currentTimeMillis();
        PART part = new PART(); // rule learner???
        part.buildClassifier(insectSoundTrain);
        ArrayList<int[]> partConfMatrix = WekaTools.confusionMatrix(part, insectSoundTest);
        double partAcc = WekaTools.accuracy(partConfMatrix);
        double partBalAcc = WekaTools.balancedAccuracy(partConfMatrix);
        accs[5].append("\t").append(partAcc);
        balAccs[5].append("\t").append(partBalAcc);
        nlls[5].append("\t").append(WekaTools.nll(part, insectSoundTest));
        System.out.print("PART(" + (System.currentTimeMillis()-startTime) + "),");

        startTime = System.currentTimeMillis();
        NaiveBayes naiveBayes = new NaiveBayes(); // With/without kernels
        naiveBayes.buildClassifier(insectSoundTrain);
        ArrayList<int[]> naiveConfMatrix = WekaTools.confusionMatrix(naiveBayes, insectSoundTest);
        double naiveAcc = WekaTools.accuracy(naiveConfMatrix);
        double naiveBalAcc = WekaTools.balancedAccuracy(naiveConfMatrix);
        accs[6].append("\t").append(naiveAcc);
        balAccs[6].append("\t").append(naiveBalAcc);
        nlls[6].append("\t").append(WekaTools.nll(naiveBayes, insectSoundTest));
        System.out.print("Naive(" + (System.currentTimeMillis()-startTime) + "),");

        startTime = System.currentTimeMillis();
        OneR oneR = new OneR(); // Holte's OneR
        oneR.buildClassifier(insectSoundTrain);
        ArrayList<int[]> oneConfMatrix = WekaTools.confusionMatrix(oneR, insectSoundTest);
        double oneRAcc = WekaTools.accuracy(oneConfMatrix);
        double oneBalAcc = WekaTools.balancedAccuracy(oneConfMatrix);
        accs[7].append("\t").append(oneRAcc);
        balAccs[7].append("\t").append(oneBalAcc);
        nlls[7].append("\t").append(WekaTools.nll(oneR, insectSoundTest));
        System.out.print("OneR(" + (System.currentTimeMillis()-startTime) + "),");

        startTime = System.currentTimeMillis();
        SMO smo = new SMO(); // SVM
        smo.buildClassifier(insectSoundTrain);
        ArrayList<int[]> smoConfMatrix = WekaTools.confusionMatrix(smo, insectSoundTest);
        double smoAcc = WekaTools.accuracy(smoConfMatrix);
        double smoBalAcc = WekaTools.balancedAccuracy(smoConfMatrix);
        accs[8].append("\t").append(smoAcc);
        balAccs[8].append("\t").append(smoBalAcc);
        nlls[8].append("\t").append(WekaTools.nll(smo, insectSoundTest));
        System.out.print("SMO(" + (System.currentTimeMillis()-startTime) + "),");

        startTime = System.currentTimeMillis();
        Logistic logistic = new Logistic(); // Logistic Regression
        logistic.buildClassifier(insectSoundTrain);
        ArrayList<int[]> logisticConfMatrix = WekaTools.confusionMatrix(logistic, insectSoundTest);
        double logisticAcc = WekaTools.accuracy(logisticConfMatrix);
        double logisticBalAcc = WekaTools.balancedAccuracy(logisticConfMatrix);
        accs[9].append("\t").append(logisticAcc);
        balAccs[9].append("\t").append(logisticBalAcc);
        nlls[9].append("\t").append(WekaTools.nll(logistic, insectSoundTest));
        System.out.print("Logistic(" + (System.currentTimeMillis()-startTime) + "),");

        startTime = System.currentTimeMillis();
        LogitBoost logitBoost = new LogitBoost();
        logitBoost.buildClassifier(insectSoundTrain);
        ArrayList<int[]> logitConfMatrix = WekaTools.confusionMatrix(logitBoost, insectSoundTest);
        double logitAcc = WekaTools.accuracy(logitConfMatrix);
        double logitBalAcc = WekaTools.balancedAccuracy(logitConfMatrix);
        accs[10].append("\t").append(logitAcc);
        balAccs[10].append("\t").append(logitBalAcc);
        nlls[10].append("\t").append(WekaTools.nll(logitBoost, insectSoundTest));
        System.out.print("Logit(" + (System.currentTimeMillis()-startTime) + "),");

        startTime = System.currentTimeMillis();
        DecisionStump decisionStump = new DecisionStump(); // For boosting?
        decisionStump.buildClassifier(insectSoundTrain);
        ArrayList<int[]> stumpConfMatrix = WekaTools.confusionMatrix(decisionStump, insectSoundTest);
        double stumpAcc = WekaTools.accuracy(stumpConfMatrix);
        double stumpBalAcc = WekaTools.balancedAccuracy(stumpConfMatrix);
        accs[11].append("\t").append(stumpAcc);
        balAccs[11].append("\t").append(stumpBalAcc);
        nlls[11].append("\t").append(WekaTools.nll(decisionStump, insectSoundTest));
        System.out.print("Stump(" + (System.currentTimeMillis()-startTime) + "),");

        // Ensemble Classifiers
        startTime = System.currentTimeMillis();
        Bagging bagging = new Bagging();
        bagging.buildClassifier(insectSoundTrain);
        ArrayList<int[]> baggingConfMatrix = WekaTools.confusionMatrix(bagging, insectSoundTest);
        double baggingAcc = WekaTools.accuracy(bagging, insectSoundTest);
        double baggingBalAcc = WekaTools.balancedAccuracy(baggingConfMatrix);
        accs[12].append("\t").append(baggingAcc);
        balAccs[12].append("\t").append(baggingBalAcc);
        nlls[12].append("\t").append(WekaTools.nll(bagging, insectSoundTest));
        System.out.print("Bag(" + (System.currentTimeMillis()-startTime) + "),");

        startTime = System.currentTimeMillis();
        RandomForest randomForest = new RandomForest();
        randomForest.buildClassifier(insectSoundTrain);
        ArrayList<int[]> ranForConfMatrix = WekaTools.confusionMatrix(randomForest, insectSoundTest);
        double ranForAcc = WekaTools.accuracy(ranForConfMatrix);
        double ranForBalAcc = WekaTools.balancedAccuracy(ranForConfMatrix);
        accs[13].append("\t").append(ranForAcc);
        balAccs[13].append("\t").append(ranForBalAcc);
        nlls[13].append("\t").append(WekaTools.nll(randomForest, insectSoundTest));
        System.out.print("RanFor(" + (System.currentTimeMillis()-startTime) + "),");

        startTime = System.currentTimeMillis();
        RotationForest rotationForest = new RotationForest();
        rotationForest.buildClassifier(insectSoundTrain);
        ArrayList<int[]> rotForConfMatrix = WekaTools.confusionMatrix(rotationForest, insectSoundTest);
        double rotForAcc = WekaTools.accuracy(rotForConfMatrix);
        double rotForBalAcc = WekaTools.balancedAccuracy(rotForConfMatrix);
        accs[14].append("\t").append(rotForAcc);
        balAccs[14].append("\t").append(rotForBalAcc);
        nlls[14].append("\t").append(WekaTools.nll(rotationForest, insectSoundTest));
        System.out.print("RotFor(" + (System.currentTimeMillis()-startTime) + "),");

        startTime = System.currentTimeMillis();
        AdaBoostM1 adaBoostM1 = new AdaBoostM1();
        adaBoostM1.buildClassifier(insectSoundTrain);
        ArrayList<int[]> adaConfMatrix = WekaTools.confusionMatrix(adaBoostM1, insectSoundTest);
        double adaAcc = WekaTools.accuracy(adaConfMatrix);
        double adaBalAcc = WekaTools.balancedAccuracy(adaConfMatrix);
        accs[15].append("\t").append(adaAcc);
        balAccs[15].append("\t").append(adaBalAcc);
        nlls[15].append("\t").append(WekaTools.nll(adaBoostM1, insectSoundTest));
        System.out.println("Ada(" + (System.currentTimeMillis()-startTime) + ")");

//        for (StringBuilder accString : accs) {
//            System.out.println(accString);
//        }
//        for (StringBuilder balAccString : balAccs) {
//            System.out.println(balAccString);
//        }
        for (StringBuilder nll : nlls) {
            System.out.println(nll);
        }
        //*/
    }
}
