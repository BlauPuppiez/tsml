package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.core.*;

import java.util.Arrays;

/**
 * A basic decision tree classifier for use in machine learning coursework (6002B).
 */
public class CourseworkTree extends AbstractClassifier {

    /** Measure to use when selecting an attribute to split the data with. */
    private AttributeSplitMeasure attSplitMeasure = new IGAttributeSplitMeasure();

    /** Maximum depth for the tree. */
    private int maxDepth = Integer.MAX_VALUE;

    /** The root node of the tree. */
    private TreeNode root;

    /**
     * Sets the option for attSplitMeasure in the classifier.
     */
    public void setOptions(String attSplitMeasure) throws Exception {
        switch (attSplitMeasure.toLowerCase()) {
            case "ig":
                setAttSplitMeasure(new IGAttributeSplitMeasure());
                break;
            case "igr":
                IGAttributeSplitMeasure igAttributeSplitMeasure = new IGAttributeSplitMeasure();
                igAttributeSplitMeasure.setUseGain(false);
                setAttSplitMeasure(igAttributeSplitMeasure);
                break;
            case "chi":
                setAttSplitMeasure(new ChiSquaredAttributeSplitMeasure());
                break;
            case "gini":
                setAttSplitMeasure(new GiniAttributeSplitMeasure());
                break;
            default:
                throw new Exception();
        }
    }

    /**
     * Sets the attribute split measure for the classifier.
     *
     * @param attSplitMeasure the split measure
     */
    public void setAttSplitMeasure(AttributeSplitMeasure attSplitMeasure) {
        this.attSplitMeasure = attSplitMeasure;
    }

    /**
     * Sets the max depth for the classifier.
     *
     * @param maxDepth the max depth
     */
    public void setMaxDepth(int maxDepth){
        this.maxDepth = maxDepth;
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        //instances
        result.setMinimumNumberInstances(2);

        return result;
    }

    /**
     * Builds a decision tree classifier.
     *
     * @param data the training data
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (data.classIndex() != data.numAttributes() - 1) {
            throw new Exception("Class attribute must be the last index.");
        }

        root = new TreeNode();
        root.buildTree(data, 0);
    }

    /**
     * Classifies a given test instance using the decision tree.
     *
     * @param instance the instance to be classified
     * @return the classification
     */
    @Override
    public double classifyInstance(Instance instance) {
        double[] probs = distributionForInstance(instance);

        int maxClass = 0;
        for (int n = 1; n < probs.length; n++) {
            if (probs[n] > probs[maxClass]) {
                maxClass = n;
            }
        }

        return maxClass;
    }

    /**
     * Computes class distribution for instance using the decision tree.
     *
     * @param instance the instance for which distribution is to be computed
     * @return the class distribution for the given instance
     */
    @Override
    public double[] distributionForInstance(Instance instance) {
        return root.distributionForInstance(instance);
    }

    /**
     * Class representing a single node in the tree.
     */
    private class TreeNode {

        /** Attribute used for splitting, if null the node is a leaf. */
        Attribute bestSplit = null;

        /** Threshold used if the bestSplit attribute is numeric **/
        double threshold;

        /** Best gain from the splitting measure if the node is not a leaf. */
        double bestGain = 0;

        /** Depth of the node in the tree. */
        int depth;

        /** The node's children if it is not a leaf. */
        TreeNode[] children;

        /** The class distribution if the node is a leaf. */
        double[] leafDistribution;

        /**
         * Recursive function for building the tree.
         * Builds a single tree node, finding the best attribute to split on using a splitting measure.
         * Splits the best attribute into multiple child tree node's if they can be made, else creates a leaf node.
         *
         * @param data Instances to build the tree node with
         * @param depth the depth of the node in the tree
         */
        void buildTree(Instances data, int depth) throws Exception {
            this.depth = depth;
            // Need to split numeric attributes before assessing the quality
            // Loop through each attribute, finding the best one.
            for (int i = 0; i < data.numAttributes() - 1; i++) {
                double gain = attSplitMeasure.computeAttributeQuality(data, data.attribute(i));

                if (gain > bestGain) {
                    bestSplit = data.attribute(i);
                    bestGain = gain;
                }
            }

            // If we found an attribute to split on, create child nodes.
            if (bestSplit != null) {
                Instances[] split;
                if (bestSplit.isNumeric()) {
                    // Calculate the threshold, so that it may be used in classification, data split etc.
                    threshold = 0.0;
                    for (Instance inst : data) {
                        threshold += inst.value(bestSplit);
                    }
                    threshold /= data.size();

                    split = attSplitMeasure.splitDataOnNumeric(data, bestSplit, threshold);
                } else {
                    split = attSplitMeasure.splitData(data, bestSplit);
                }
                children = new TreeNode[split.length];

                // Create a child for each value in the selected attribute, and determine whether it is a leaf or not.
                for (int i = 0; i < children.length; i++) {
                    children[i] = new TreeNode();

                    boolean leaf = split[i].numDistinctValues(data.classIndex()) == 1 || depth + 1 == maxDepth;

                    if (split[i].isEmpty()) {
                        children[i].buildLeaf(data, depth + 1);
                    } else if (leaf) {
                        children[i].buildLeaf(split[i], depth + 1);
                    } else {
                        children[i].buildTree(split[i], depth + 1);
                    }
                }
            // Else turn this node into a leaf node.
            } else {
                leafDistribution = classDistribution(data);
            }
        }

        /**
         * Builds a leaf node for the tree, setting the depth and recording the class distribution of the remaining
         * instances.
         *
         * @param data remaining Instances to build the leafs class distribution
         * @param depth the depth of the node in the tree
         */
        void buildLeaf(Instances data, int depth) {
            this.depth = depth;
            leafDistribution = classDistribution(data);
        }

        /**
         * Recursive function traversing node's of the tree until a leaf is found. Returns the leafs class distribution.
         *
         * @return the class distribution of the first leaf node
         */
        double[] distributionForInstance(Instance inst) {
            // If the node is a leaf return the distribution, else select the next node based on the best attributes
            // value.
            if (bestSplit == null) {
                return leafDistribution;
            } else {
                // Check for numeric attribute, if it is use the threshold on the numeric value.
                if (bestSplit.isNumeric()) {
                    if (inst.value(bestSplit) < threshold) {
                        return children[0].distributionForInstance(inst);
                    } else {
                        return children[1].distributionForInstance(inst);
                    }
                } else {
                    return children[(int) inst.value(bestSplit)].distributionForInstance(inst);
                }
            }
        }

        /**
         * Returns the normalised version of the input array with values summing to 1.
         *
         * @return the class distribution as an array
         */
        double[] classDistribution(Instances data) {
            double[] distribution = new double[data.numClasses()];
            for (Instance inst : data) {
                distribution[(int) inst.classValue()]++;
            }

            double sum = 0;
            for (double d : distribution) {
                sum += d;
            }

            if (sum != 0) {
                for (int i = 0; i < distribution.length; i++) {
                    distribution[i] = distribution[i] / sum;
                }
            }

            return distribution;
        }

        /**
         * Summarises the tree node into a String.
         *
         * @return the summarised node as a String
         */
        @Override
        public String toString() {
            String str;
            if (bestSplit == null) {
                str = "Leaf," + Arrays.toString(leafDistribution) + "," + depth;
            } else {
                str = bestSplit.name() + "," + bestGain + "," + depth;
            }
            return str;
        }
    }

    /**
     * Main method.
     *
     * @param args the options for the classifier main
     */
    public static void main(String[] args) throws Exception {
        Instances optdigitsData = WekaTools.loadLocalClassificationData("optdigits.arff");
        Instances chinatownData = WekaTools.loadLocalClassificationData("Chinatown.arff");

        // Random split 0.8 : 0.2 (training : test) split
        assert optdigitsData != null;
        assert chinatownData != null;
        Instances[] optidigitsSplit = WekaTools.splitDataRandom(optdigitsData, 0.8);
        Instances[] chinatownSplit = WekaTools.splitDataRandom(chinatownData, 0.8);

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

        CourseworkTree chinatownCWTreeIG = new CourseworkTree();
        chinatownCWTreeIG.buildClassifier(chinatownSplit[0]);
        double chinatownIGAcc = WekaTools.accuracy(chinatownCWTreeIG, chinatownSplit[1]);
        System.out.println("DT using measure Information Gain on Chinatown problem has test accuracy = " + chinatownIGAcc);

        courseworkTreeIGR.buildClassifier(chinatownSplit[0]);
        igrAcc = WekaTools.accuracy(courseworkTreeIGR, chinatownSplit[1]);
        System.out.println("DT using measure Information Gain Ratio on Chinatown problem has test accuracy = " + igrAcc);

        courseworkTreeChi.buildClassifier(chinatownSplit[0]);
        chiAcc = WekaTools.accuracy(courseworkTreeChi, chinatownSplit[1]);
        System.out.println("DT using measure Chi-Squared on Chinatown problem has test accuracy = " + chiAcc);

        courseworkTreeGini.buildClassifier(chinatownSplit[0]);
        giniAcc = WekaTools.accuracy(courseworkTreeGini, chinatownSplit[1]);
        System.out.println("DT using measure Gini on Chinatown problem has test accuracy = " + giniAcc);
    }
}