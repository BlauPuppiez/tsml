package ml_6002b_coursework;

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
        System.out.println(data.numAttributes()); // Attribute Count
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
    public static double round(double value) {
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
        //*/

        Instances insectWingbeatTrain = WekaTools.loadLocalClassificationData("InsectWingbeatTRAIN.arff");
        Instances insectWingbeatTest = WekaTools.loadLocalClassificationData("InsectWingbeatTEST.arff");

        // Summary of datasets:
        /*
        System.out.println("Summary of Datasets");
        System.out.println("Discrete Datasets " + discreteCount);
        for (int discreteIt = 0; discreteIt < discreteCount; discreteIt++) {
//            System.out.println(discreteDatasetNames[discreteIt]);
//            System.out.println(discreteData[discreteIt].numAttributes()); // Attribute Count
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
//            System.out.println(continuousDatasetNames[continuousIt]);
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

        // Specified Dataset:
//        System.out.println("InsectWingbeatTrain Summary");
//        SummariseData(insectWingbeatTrain);
//        System.out.println("InsectWingbeatTest Summary");
//        SummariseData(insectWingbeatTest);

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

        for (int discreteIt = 0; discreteIt < discreteCount; discreteIt++) {
            Instances dataset = discreteData[discreteIt];
            Instances[] dataSplit = WekaTools.splitData(dataset, 0.8);

            //System.out.println("Evaluation of " + discreteDatasetNames[discreteIt] + ":");
            columns.append(discreteDatasetNames[discreteIt]).append("\t");

            CourseworkTree courseworkTree = new CourseworkTree();
            courseworkTree.buildClassifier(dataSplit[0]);
            double ctAcc = WekaTools.accuracy(courseworkTree, dataSplit[1]);
            accs[0].append(ctAcc).append("\t");

            Id3 id3 = new Id3();
            id3.buildClassifier(dataSplit[0]);
            double id3Acc = WekaTools.accuracy(id3, dataSplit[1]);
            accs[1].append(id3Acc).append("\t");

            J48 j48 = new J48();
            j48.buildClassifier(dataSplit[0]);
            double j48Acc = WekaTools.accuracy(j48, dataSplit[1]);
            accs[2].append(j48Acc).append("\t");
        }
        System.out.println(columns);
        System.out.println(accs[0]);
        System.out.println(accs[1]);
        System.out.println(accs[2]);
        //*/

        // And for Continuous Datasets
        /*
        System.out.println("Evaluation of Continuous Datasets");
        StringBuilder columns = new StringBuilder();
        StringBuilder[] accs = new StringBuilder[3];
        accs[0] = new StringBuilder();
        accs[1] = new StringBuilder();
        accs[2] = new StringBuilder();

        for (int continuousIt = 0; continuousIt < continuousCount; continuousIt++) {
            Instances dataset = continuousData[continuousIt];
            Instances[] dataSplit = WekaTools.splitData(dataset, 0.8);

            //System.out.println("Evaluation of " + continuousDatasetNames[continuousIt] + ":");
            columns.append(continuousDatasetNames[continuousIt]).append("\t");

            CourseworkTree courseworkTree = new CourseworkTree();
            courseworkTree.buildClassifier(dataSplit[0]);
            double ctAcc = WekaTools.accuracy(courseworkTree, dataSplit[1]);
            accs[0].append(ctAcc).append("\t");

            SimpleCart simpleCart = new SimpleCart();
            simpleCart.buildClassifier(dataSplit[0]);
            double cartAcc = WekaTools.accuracy(simpleCart, dataSplit[1]);
            accs[1].append(cartAcc).append("\t");

            J48 j48 = new J48();
            j48.buildClassifier(dataSplit[0]);
            double j48Acc = WekaTools.accuracy(j48, dataSplit[1]);
            accs[2].append(j48Acc).append("\t");
        }
        System.out.println(columns);
        System.out.println(accs[0]);
        System.out.println(accs[1]);
        System.out.println(accs[2]);
        //*/

        // PART 2)
        /*
        TreeEnsemble treeEnsemble = new TreeEnsemble();

        StringBuilder columns = new StringBuilder("Accuracy");
        StringBuilder[] accs = new StringBuilder[8];
        accs[0] = new StringBuilder("1, 100%, false");
        accs[1] = new StringBuilder("1, 50%, false");
        accs[2] = new StringBuilder("5, 100%, false");
        accs[3] = new StringBuilder("5, 50%, false");
        accs[4] = new StringBuilder("5, 100%, true");
        accs[5] = new StringBuilder("5, 50%, true");
        accs[6] = new StringBuilder("50, 50%, false");
        accs[7] = new StringBuilder("50, 50%, true");

        for (int discreteIt = 0; discreteIt < discreteCount; discreteIt++) {
            Instances[] trainTestSplit = WekaTools.splitData(discreteData[discreteIt], 0.8);

            columns.append("\t").append(discreteDatasetNames[discreteIt]);

//            treeEnsemble.setNumTrees(1);
//            treeEnsemble.setAttProp(1.0);
//            treeEnsemble.setAverageDistributions(false);
//            treeEnsemble.buildClassifier(trainTestSplit[0]);
//            accs[0].append("\t").append(WekaTools.accuracy(treeEnsemble, trainTestSplit[1]));
//
//
//            treeEnsemble.setNumTrees(1);
//            treeEnsemble.setAttProp(0.5);
//            treeEnsemble.setAverageDistributions(false);
//            treeEnsemble.buildClassifier(trainTestSplit[0]);
//            accs[1].append("\t").append(WekaTools.accuracy(treeEnsemble, trainTestSplit[1]));
//
//            treeEnsemble.setNumTrees(5);
//            treeEnsemble.setAttProp(1.0);
//            treeEnsemble.setAverageDistributions(false);
//            treeEnsemble.buildClassifier(trainTestSplit[0]);
//            accs[2].append("\t").append(WekaTools.accuracy(treeEnsemble, trainTestSplit[1]));
//
//            treeEnsemble.setNumTrees(5);
//            treeEnsemble.setAttProp(0.5);
//            treeEnsemble.setAverageDistributions(false);
//            treeEnsemble.buildClassifier(trainTestSplit[0]);
//            accs[3].append("\t").append(WekaTools.accuracy(treeEnsemble, trainTestSplit[1]));

//            treeEnsemble.setNumTrees(5);
//            treeEnsemble.setAttProp(1.0);
//            treeEnsemble.setAverageDistributions(true);
//            treeEnsemble.buildClassifier(trainTestSplit[0]);
//            accs[4].append("\t").append(WekaTools.accuracy(treeEnsemble, trainTestSplit[1]));
//
//            treeEnsemble.setNumTrees(5);
//            treeEnsemble.setAttProp(0.5);
//            treeEnsemble.setAverageDistributions(true);
//            treeEnsemble.buildClassifier(trainTestSplit[0]);
//            accs[5].append("\t").append(WekaTools.accuracy(treeEnsemble, trainTestSplit[1]));

            treeEnsemble.setNumTrees(50);
            treeEnsemble.setAttProp(1.0);
            treeEnsemble.setAverageDistributions(false);
            treeEnsemble.buildClassifier(trainTestSplit[0]);
            accs[6].append("\t").append(WekaTools.accuracy(treeEnsemble, trainTestSplit[1]));

            treeEnsemble.setNumTrees(50);
            treeEnsemble.setAttProp(1.0);
            treeEnsemble.setAverageDistributions(true);
            treeEnsemble.buildClassifier(trainTestSplit[0]);
            accs[7].append("\t").append(WekaTools.accuracy(treeEnsemble, trainTestSplit[1]));
        }
        System.out.println(columns);
        for (StringBuilder accString : accs) {
            System.out.println(accString);
        }
        //*/

        // Part 3)
        // Contains times for each classifier on each problem
        // This was to check for classifications that took too long and ignore them
        StringBuilder columns = new StringBuilder("Measure");

        StringBuilder[] accs = new StringBuilder[16];
        accs[0] = new StringBuilder("TreeEnsemble");
        accs[1] = new StringBuilder("IBk");
        accs[2] = new StringBuilder("J48");
        accs[3] = new StringBuilder("PART");
        accs[4] = new StringBuilder("NaiveBayes");
        accs[5] = new StringBuilder("OneR");
        accs[6] = new StringBuilder("SMO");
        accs[7] = new StringBuilder("Logistic");
        accs[8] = new StringBuilder("LogitBoost");
        accs[9] = new StringBuilder("DecisionStump");
        accs[10] = new StringBuilder("Bagging");
        accs[11] = new StringBuilder("RandomForest");
        accs[12] = new StringBuilder("RotationForest");
        accs[13] = new StringBuilder("AdaBoostM1");
        accs[14] = new StringBuilder("Voting");
        accs[15] = new StringBuilder("Stacking");

        StringBuilder[] balAccs = new StringBuilder[16];
        balAccs[0] = new StringBuilder("TreeEnsemble");
        balAccs[1] = new StringBuilder("IBk");
        balAccs[2] = new StringBuilder("J48");
        balAccs[3] = new StringBuilder("PART");
        balAccs[4] = new StringBuilder("NaiveBayes");
        balAccs[5] = new StringBuilder("OneR");
        balAccs[6] = new StringBuilder("SMO");
        balAccs[7] = new StringBuilder("Logistic");
        balAccs[8] = new StringBuilder("LogitBoost");
        balAccs[9] = new StringBuilder("DecisionStump");
        balAccs[10] = new StringBuilder("Bagging");
        balAccs[11] = new StringBuilder("RandomForest");
        balAccs[12] = new StringBuilder("RotationForest");
        balAccs[13] = new StringBuilder("AdaBoostM1");
        balAccs[14] = new StringBuilder("Voting");
        balAccs[15] = new StringBuilder("Stacking");

        long startTime;
        ///* Discrete Datasets
        for (int discreteIt = 0; discreteIt < discreteCount; discreteIt++) {
            Instances[] trainTestSplit = WekaTools.splitData(discreteData[discreteIt], 0.8);

            columns.append("\t").append(discreteDatasetNames[discreteIt]);
            System.out.println(discreteDatasetNames[discreteIt]);

//            startTime = System.currentTimeMillis();
//            TreeEnsemble treeEnsemble = new TreeEnsemble();
//            treeEnsemble.setNumTrees(10);
//            treeEnsemble.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> confMatrix = WekaTools.confusionMatrix(treeEnsemble, trainTestSplit[1]);
////            for (int[] classLine : confMatrix) { // Debug: confusion matrix
////                System.out.println(Arrays.toString(classLine));
////            }
//            double acc = WekaTools.accuracy(confMatrix);
//            double balAcc = WekaTools.balancedAccuracy(confMatrix);
//            accs[0].append("\t").append(acc);
//            balAccs[0].append("\t").append(balAcc);
//            System.out.print("TreeEnsemble(" + (System.currentTimeMillis()-startTime) + "),");
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
//            accs[1].append("\t").append(ibkAcc);
//            balAccs[1].append("\t").append(ibkBalAcc);
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
//            accs[3].append("\t").append(partAcc);
//            balAccs[3].append("\t").append(partBalAcc);
//            System.out.print("PART(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            NaiveBayes naiveBayes = new NaiveBayes(); // With/without kernels
//            naiveBayes.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> naiveConfMatrix = WekaTools.confusionMatrix(naiveBayes, trainTestSplit[1]);
//            double naiveAcc = WekaTools.accuracy(naiveConfMatrix);
//            double naiveBalAcc = WekaTools.balancedAccuracy(naiveConfMatrix);
//            accs[4].append("\t").append(naiveAcc);
//            balAccs[4].append("\t").append(naiveBalAcc);
//            System.out.print("Naive(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            OneR oneR = new OneR(); // Holte's OneR
//            oneR.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> oneConfMatrix = WekaTools.confusionMatrix(oneR, trainTestSplit[1]);
//            double oneRAcc = WekaTools.accuracy(oneConfMatrix);
//            double oneBalAcc = WekaTools.balancedAccuracy(oneConfMatrix);
//            accs[5].append("\t").append(oneRAcc);
//            balAccs[5].append("\t").append(oneBalAcc);
//            System.out.print("OneR(" + (System.currentTimeMillis()-startTime) + "),");

            startTime = System.currentTimeMillis();
            SMO smo = new SMO(); // SVM
            smo.buildClassifier(trainTestSplit[0]);
            ArrayList<int[]> smoConfMatrix = WekaTools.confusionMatrix(smo, trainTestSplit[1]);
            double smoAcc = WekaTools.accuracy(smoConfMatrix);
            double smoBalAcc = WekaTools.balancedAccuracy(smoConfMatrix);
            accs[6].append("\t").append(smoAcc);
            balAccs[6].append("\t").append(smoBalAcc);
            System.out.print("SMO(" + (System.currentTimeMillis()-startTime) + "),");

            startTime = System.currentTimeMillis();
            Logistic logistic = new Logistic(); // Logistic Regression
            logistic.buildClassifier(trainTestSplit[0]);
            ArrayList<int[]> logisticConfMatrix = WekaTools.confusionMatrix(logistic, trainTestSplit[1]);
            double logisticAcc = WekaTools.accuracy(logisticConfMatrix);
            double logisticBalAcc = WekaTools.balancedAccuracy(logisticConfMatrix);
            accs[7].append("\t").append(logisticAcc);
            balAccs[7].append("\t").append(logisticBalAcc);
            System.out.print("Logistic(" + (System.currentTimeMillis()-startTime) + "),");

//            startTime = System.currentTimeMillis();
//            LogitBoost logitBoost = new LogitBoost();
//            logitBoost.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> logitConfMatrix = WekaTools.confusionMatrix(logitBoost, trainTestSplit[1]);
//            double logitAcc = WekaTools.accuracy(logitConfMatrix);
//            double logitBalAcc = WekaTools.balancedAccuracy(logitConfMatrix);
//            accs[8].append("\t").append(logitAcc);
//            balAccs[8].append("\t").append(logitBalAcc);
//            System.out.print("Logit(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            DecisionStump decisionStump = new DecisionStump(); // For boosting?
//            decisionStump.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> stumpConfMatrix = WekaTools.confusionMatrix(decisionStump, trainTestSplit[1]);
//            double stumpAcc = WekaTools.accuracy(stumpConfMatrix);
//            double stumpBalAcc = WekaTools.balancedAccuracy(stumpConfMatrix);
//            accs[9].append("\t").append(stumpAcc);
//            balAccs[9].append("\t").append(stumpBalAcc);
//            System.out.print("Stump(" + (System.currentTimeMillis()-startTime) + "),");
//
//            // Ensemble Classifiers
//            startTime = System.currentTimeMillis();
//            Bagging bagging = new Bagging();
//            bagging.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> baggingConfMatrix = WekaTools.confusionMatrix(bagging, trainTestSplit[1]);
//            double baggingAcc = WekaTools.accuracy(bagging, trainTestSplit[1]);
//            double baggingBalAcc = WekaTools.balancedAccuracy(baggingConfMatrix);
//            accs[10].append("\t").append(baggingAcc);
//            balAccs[10].append("\t").append(baggingBalAcc);
//            System.out.print("Bag(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            RandomForest randomForest = new RandomForest();
//            randomForest.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> ranForConfMatrix = WekaTools.confusionMatrix(randomForest, trainTestSplit[1]);
//            double ranForAcc = WekaTools.accuracy(ranForConfMatrix);
//            double ranForBalAcc = WekaTools.balancedAccuracy(ranForConfMatrix);
//            accs[11].append("\t").append(ranForAcc);
//            balAccs[11].append("\t").append(ranForBalAcc);
//            System.out.print("RanFor(" + (System.currentTimeMillis()-startTime) + "),");

            startTime = System.currentTimeMillis();
            RotationForest rotationForest = new RotationForest();
            rotationForest.buildClassifier(trainTestSplit[0]);
            ArrayList<int[]> rotForConfMatrix = WekaTools.confusionMatrix(rotationForest, trainTestSplit[1]);
            double rotForAcc = WekaTools.accuracy(rotForConfMatrix);
            double rotForBalAcc = WekaTools.balancedAccuracy(rotForConfMatrix);
            accs[12].append("\t").append(rotForAcc);
            accs[12].append("\t").append(rotForBalAcc);
            System.out.print("RotFor(" + (System.currentTimeMillis()-startTime) + "),");

//            startTime = System.currentTimeMillis();
//            AdaBoostM1 adaBoostM1 = new AdaBoostM1();
//            adaBoostM1.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> adaConfMatrix = WekaTools.confusionMatrix(adaBoostM1, trainTestSplit[1]);
//            double adaAcc = WekaTools.accuracy(adaConfMatrix);
//            double adaBalAcc = WekaTools.balancedAccuracy(adaConfMatrix);
//            accs[13].append("\t").append(adaAcc);
//            balAccs[13].append("\t").append(adaBalAcc);
//            System.out.print("Ada(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            Vote vote = new Vote();
//            vote.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> voteConfMatrix = WekaTools.confusionMatrix(vote, trainTestSplit[1]);
//            double voteAcc = WekaTools.accuracy(voteConfMatrix);
//            double voteBalAcc = WekaTools.balancedAccuracy(voteConfMatrix);
//            accs[14].append("\t").append(voteAcc);
//            balAccs[14].append("\t").append(voteBalAcc);
//            System.out.print("Vote(" + (System.currentTimeMillis()-startTime) + "),");
//
//            startTime = System.currentTimeMillis();
//            Stacking stacking = new Stacking();
//            stacking.buildClassifier(trainTestSplit[0]);
//            ArrayList<int[]> stackConfMatrix = WekaTools.confusionMatrix(stacking, trainTestSplit[1]);
//            double stackAcc = WekaTools.accuracy(stackConfMatrix);
//            double stackBalAcc = WekaTools.balancedAccuracy(stackConfMatrix);
//            accs[15].append("\t").append(stackAcc);
//            balAccs[15].append("\t").append(stackBalAcc);
//            System.out.print("Stack(" + (System.currentTimeMillis()-startTime) + "),");

            System.out.println();
        }
        System.out.println(columns);
//        for (StringBuilder accString : accs) {
//            System.out.println(accString);
//        }
        for (StringBuilder balAccString : balAccs) {
            System.out.println(balAccString);
        }
        //*/

        /* Continuous datasets
        for (int continuousIt = 0; continuousIt < continuousCount; continuousIt++) {
            Instances[] trainTestSplit = WekaTools.splitData(continuousData[continuousIt], 0.8);

            columns.append("\t").append(continuousDatasetNames[continuousIt]);
            System.out.println(continuousDatasetNames[continuousIt]);

            startTime = System.currentTimeMillis();
            TreeEnsemble treeEnsemble = new TreeEnsemble();
            treeEnsemble.setNumTrees(10);
            treeEnsemble.buildClassifier(trainTestSplit[0]);
            ArrayList<int[]> confMatrix = WekaTools.confusionMatrix(treeEnsemble, trainTestSplit[1]);
            double acc = WekaTools.accuracy(confMatrix);
            double balAcc = WekaTools.balancedAccuracy(confMatrix);
            accs[0].append("\t").append(acc);
            balAccs[0].append("\t").append(balAcc);
            System.out.print("TreeEnsemble(" + (System.currentTimeMillis()-startTime) + "),");

            startTime = System.currentTimeMillis();
            IBk ibk = new IBk(); // k-NN
            ibk.buildClassifier(trainTestSplit[0]);
            ArrayList<int[]> ibkConfMatrix = WekaTools.confusionMatrix(ibk, trainTestSplit[1]);
            double ibkAcc = WekaTools.accuracy(ibkConfMatrix);
            double ibkBalAcc = WekaTools.balancedAccuracy(ibkConfMatrix);
            accs[1].append("\t").append(ibkAcc);
            balAccs[1].append("\t").append(ibkBalAcc);
            System.out.print("IBk(" + (System.currentTimeMillis()-startTime) + "),");

            startTime = System.currentTimeMillis();
            J48 j48 = new J48(); // C4.5 decision tree(s)
            j48.buildClassifier(trainTestSplit[0]);
            ArrayList<int[]> j48ConfMatrix = WekaTools.confusionMatrix(j48, trainTestSplit[1]);
            double j48Acc = WekaTools.accuracy(j48ConfMatrix);
            double j48BalAcc = WekaTools.balancedAccuracy(j48ConfMatrix);
            accs[2].append("\t").append(j48Acc);
            balAccs[2].append("\t").append(j48BalAcc);
            System.out.print("J48(" + (System.currentTimeMillis()-startTime) + "),");

            startTime = System.currentTimeMillis();
            PART part = new PART(); // rule learner???
            part.buildClassifier(trainTestSplit[0]);
            ArrayList<int[]> partConfMatrix = WekaTools.confusionMatrix(part, trainTestSplit[1]);
            double partAcc = WekaTools.accuracy(partConfMatrix);
            double partBalAcc = WekaTools.balancedAccuracy(partConfMatrix);
            accs[3].append("\t").append(partAcc);
            balAccs[3].append("\t").append(partBalAcc);
            System.out.print("PART(" + (System.currentTimeMillis()-startTime) + "),");

            startTime = System.currentTimeMillis();
            NaiveBayes naiveBayes = new NaiveBayes(); // With/without kernels
            naiveBayes.buildClassifier(trainTestSplit[0]);
            ArrayList<int[]> naiveConfMatrix = WekaTools.confusionMatrix(naiveBayes, trainTestSplit[1]);
            double naiveAcc = WekaTools.accuracy(naiveConfMatrix);
            double naiveBalAcc = WekaTools.balancedAccuracy(naiveConfMatrix);
            accs[4].append("\t").append(naiveAcc);
            balAccs[4].append("\t").append(naiveBalAcc);
            System.out.print("Naive(" + (System.currentTimeMillis()-startTime) + "),");

            startTime = System.currentTimeMillis();
            OneR oneR = new OneR(); // Holte's OneR
            oneR.buildClassifier(trainTestSplit[0]);
            ArrayList<int[]> oneConfMatrix = WekaTools.confusionMatrix(oneR, trainTestSplit[1]);
            double oneRAcc = WekaTools.accuracy(oneConfMatrix);
            double oneBalAcc = WekaTools.balancedAccuracy(oneConfMatrix);
            accs[5].append("\t").append(oneRAcc);
            balAccs[5].append("\t").append(oneBalAcc);
            System.out.print("OneR(" + (System.currentTimeMillis()-startTime) + "),");

            startTime = System.currentTimeMillis();
            SMO smo = new SMO(); // SVM
            smo.buildClassifier(trainTestSplit[0]);
            ArrayList<int[]> smoConfMatrix = WekaTools.confusionMatrix(smo, trainTestSplit[1]);
            double smoAcc = WekaTools.accuracy(smoConfMatrix);
            double smoBalAcc = WekaTools.balancedAccuracy(smoConfMatrix);
            accs[6].append("\t").append(smoAcc);
            balAccs[6].append("\t").append(smoBalAcc);
            System.out.print("SMO(" + (System.currentTimeMillis()-startTime) + "),");

            startTime = System.currentTimeMillis();
            Logistic logistic = new Logistic(); // Logistic Regression
            logistic.buildClassifier(trainTestSplit[0]);
            ArrayList<int[]> logisticConfMatrix = WekaTools.confusionMatrix(logistic, trainTestSplit[1]);
            double logisticAcc = WekaTools.accuracy(logisticConfMatrix);
            double logisticBalAcc = WekaTools.balancedAccuracy(logisticConfMatrix);
            accs[7].append("\t").append(logisticAcc);
            balAccs[7].append("\t").append(logisticBalAcc);
            System.out.print("Logistic(" + (System.currentTimeMillis()-startTime) + "),");

            startTime = System.currentTimeMillis();
            LogitBoost logitBoost = new LogitBoost();
            logitBoost.buildClassifier(trainTestSplit[0]);
            ArrayList<int[]> logitConfMatrix = WekaTools.confusionMatrix(logitBoost, trainTestSplit[1]);
            double logitAcc = WekaTools.accuracy(logitConfMatrix);
            double logitBalAcc = WekaTools.balancedAccuracy(logitConfMatrix);
            accs[8].append("\t").append(logitAcc);
            balAccs[8].append("\t").append(logitBalAcc);
            System.out.print("Logit(" + (System.currentTimeMillis()-startTime) + "),");

            startTime = System.currentTimeMillis();
            DecisionStump decisionStump = new DecisionStump(); // For boosting?
            decisionStump.buildClassifier(trainTestSplit[0]);
            ArrayList<int[]> stumpConfMatrix = WekaTools.confusionMatrix(decisionStump, trainTestSplit[1]);
            double stumpAcc = WekaTools.accuracy(stumpConfMatrix);
            double stumpBalAcc = WekaTools.balancedAccuracy(stumpConfMatrix);
            accs[9].append("\t").append(stumpAcc);
            balAccs[9].append("\t").append(stumpBalAcc);
            System.out.print("Stump(" + (System.currentTimeMillis()-startTime) + "),");

            // Ensemble Classifiers
            startTime = System.currentTimeMillis();
            Bagging bagging = new Bagging();
            bagging.buildClassifier(trainTestSplit[0]);
            ArrayList<int[]> baggingConfMatrix = WekaTools.confusionMatrix(bagging, trainTestSplit[1]);
            double baggingAcc = WekaTools.accuracy(bagging, trainTestSplit[1]);
            double baggingBalAcc = WekaTools.balancedAccuracy(baggingConfMatrix);
            accs[10].append("\t").append(baggingAcc);
            balAccs[10].append("\t").append(baggingBalAcc);
            System.out.print("Bag(" + (System.currentTimeMillis()-startTime) + "),");

            startTime = System.currentTimeMillis();
            RandomForest randomForest = new RandomForest();
            randomForest.buildClassifier(trainTestSplit[0]);
            ArrayList<int[]> ranForConfMatrix = WekaTools.confusionMatrix(randomForest, trainTestSplit[1]);
            double ranForAcc = WekaTools.accuracy(ranForConfMatrix);
            double ranForBalAcc = WekaTools.balancedAccuracy(ranForConfMatrix);
            accs[11].append("\t").append(ranForAcc);
            balAccs[11].append("\t").append(ranForBalAcc);
            System.out.print("RanFor(" + (System.currentTimeMillis()-startTime) + "),");

            startTime = System.currentTimeMillis();
            RotationForest rotationForest = new RotationForest();
            rotationForest.buildClassifier(trainTestSplit[0]);
            ArrayList<int[]> rotForConfMatrix = WekaTools.confusionMatrix(rotationForest, trainTestSplit[1]);
            double rotForAcc = WekaTools.accuracy(rotForConfMatrix);
            double rotForBalAcc = WekaTools.balancedAccuracy(rotForConfMatrix);
            accs[12].append("\t").append(rotForAcc);
            accs[12].append("\t").append(rotForBalAcc);
            System.out.print("RotFor(" + (System.currentTimeMillis()-startTime) + "),");

            startTime = System.currentTimeMillis();
            AdaBoostM1 adaBoostM1 = new AdaBoostM1();
            adaBoostM1.buildClassifier(trainTestSplit[0]);
            ArrayList<int[]> adaConfMatrix = WekaTools.confusionMatrix(adaBoostM1, trainTestSplit[1]);
            double adaAcc = WekaTools.accuracy(adaConfMatrix);
            double adaBalAcc = WekaTools.balancedAccuracy(adaConfMatrix);
            accs[13].append("\t").append(adaAcc);
            balAccs[13].append("\t").append(adaBalAcc);
            System.out.print("Ada(" + (System.currentTimeMillis()-startTime) + "),");

            startTime = System.currentTimeMillis();
            Vote vote = new Vote();
            vote.buildClassifier(trainTestSplit[0]);
            ArrayList<int[]> voteConfMatrix = WekaTools.confusionMatrix(vote, trainTestSplit[1]);
            double voteAcc = WekaTools.accuracy(voteConfMatrix);
            double voteBalAcc = WekaTools.balancedAccuracy(voteConfMatrix);
            accs[14].append("\t").append(voteAcc);
            balAccs[14].append("\t").append(voteBalAcc);
            System.out.print("Vote(" + (System.currentTimeMillis()-startTime) + "),");

            startTime = System.currentTimeMillis();
            Stacking stacking = new Stacking();
            stacking.buildClassifier(trainTestSplit[0]);
            ArrayList<int[]> stackConfMatrix = WekaTools.confusionMatrix(stacking, trainTestSplit[1]);
            double stackAcc = WekaTools.accuracy(stackConfMatrix);
            double stackBalAcc = WekaTools.balancedAccuracy(stackConfMatrix);
            accs[15].append("\t").append(stackAcc);
            balAccs[15].append("\t").append(stackBalAcc);
            System.out.print("Stack(" + (System.currentTimeMillis()-startTime) + "),");

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
}
