package ml_6002b_coursework;

import weka.core.Instances;

public class DecisionTreeCoursework {

    public static void main(String[] args) throws Exception {
        Instances optdigitsData = WekaTools.loadLocalClassificationData("optdigits.arff");
        Instances chinatownData = WekaTools.loadLocalClassificationData("Chinatown.arff");

        // Random split 0.8 : 0.2 (training : test) split
        assert optdigitsData != null;
        assert chinatownData != null;
        Instances[] optidigitsSplit = WekaTools.splitDataRandom(optdigitsData, 0.8);
        Instances[] chinatownSplit = WekaTools.splitDataRandom(chinatownData, 0.8);
        WekaTools.convertNumericToNominalBinary(chinatownData);

        CourseworkTree courseworkTreeIG = new CourseworkTree();
        courseworkTreeIG.setOptions("IG");
        courseworkTreeIG.buildClassifier(optidigitsSplit[0]);
        double igAcc = WekaTools.accuracy(courseworkTreeIG, optidigitsSplit[1]);
        System.out.println("DT using measure Information Gain on optdigits problem has test accuracy = " + igAcc);

        CourseworkTree courseworkTreeIGR = new CourseworkTree();
        courseworkTreeIGR.setOptions("IGR");
        courseworkTreeIGR.buildClassifier(optidigitsSplit[0]);
        double igrAcc = WekaTools.accuracy(courseworkTreeIGR, optidigitsSplit[1]);
        System.out.println("DT using measure Information Gain Ratio on optdigits problem has test accuracy = " + igrAcc);

        CourseworkTree courseworkTreeChi = new CourseworkTree();
        courseworkTreeChi.setOptions("chi");
        courseworkTreeChi.buildClassifier(optidigitsSplit[0]);
        double chiAcc = WekaTools.accuracy(courseworkTreeChi, optidigitsSplit[1]);
        System.out.println("DT using measure Chi-Squared on optdigits problem has test accuracy = " + chiAcc);

        CourseworkTree courseworkTreeGini = new CourseworkTree();
        courseworkTreeGini.setOptions("gini");
        courseworkTreeGini.buildClassifier(optidigitsSplit[0]);
        double giniAcc = WekaTools.accuracy(courseworkTreeGini, optidigitsSplit[1]);
        System.out.println("DT using measure Gini on optdigits problem has test accuracy = " + giniAcc);

        courseworkTreeIG.buildClassifier(chinatownSplit[0]);
        igAcc = WekaTools.accuracy(courseworkTreeIG, chinatownSplit[1]);
        System.out.println("DT using measure Information Gain on optdigits problem has test accuracy = " + igAcc);

        courseworkTreeIGR.buildClassifier(chinatownSplit[0]);
        igrAcc = WekaTools.accuracy(courseworkTreeIGR, chinatownSplit[1]);
        System.out.println("DT using measure Information Gain Ratio on optdigits problem has test accuracy = " + igrAcc);

        courseworkTreeChi.buildClassifier(chinatownSplit[0]);
        chiAcc = WekaTools.accuracy(courseworkTreeChi, chinatownSplit[1]);
        System.out.println("DT using measure Chi-Squared on optdigits problem has test accuracy = " + chiAcc);

        courseworkTreeGini.buildClassifier(chinatownSplit[0]);
        giniAcc = WekaTools.accuracy(courseworkTreeGini, chinatownSplit[1]);
        System.out.println("DT using measure Gini on optdigits problem has test accuracy = " + giniAcc);
    }
}
