package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class TreeEnsemble extends AbstractClassifier {
    private CourseworkTree[] ensemble;

    /** Attribute split measure */
    AttributeSplitMeasure attributeSplitMeasure = new IGAttributeSplitMeasure();

    /** Selected attributes for each classifier in the ensemble */
    private Attribute[][] selectedAttributes;
    private ArrayList<Integer>[] selectedAttributeIndexes;
    private ArrayList<Integer>[] removeAttributeIndexes;

    /** Number of trees in the ensemble */
    private int numTrees = 50;

    /** Proportion of attributes to use in each classifier, does so without replacement */
    private double attProp = 0.5;

    /** Use average of distributions when classifying a new instance, default uses majority voting (false) */
    private boolean averageDistributions = false;

    /** Random seed value for reproducibility **/
    private long seed = 0;

    /**
     * Set the number of trees, must be a minimum of at least 1.
     * @param numTrees number of trees
     */
    public void setNumTrees(int numTrees) {
        if (numTrees >= 1) {
            this.numTrees = numTrees;
        }
    }

    /**
     * Set the proportion of attributes to keep when building the ensemble.
     * Proportion must be in range 0 < prop <= 1.
     * @param attProp proportion of attributes to keep
     */
    public void setAttProp(double attProp) {
        if (!(attProp <= 0 || attProp > 1)) {
            this.attProp = attProp;
        }
    }

    /**
     * Set the seed for random attribute selection.
     * @param seed long value for the random seed
     */
    public void setSeed(long seed) {
        this.seed = seed;
    }

    public void setAverageDistributions(boolean averageDistributions) {
        this.averageDistributions = averageDistributions;
    }

    public void setAttributeSplitMeasure(String attributeSplitMeasure) throws Exception {
        switch (attributeSplitMeasure.toLowerCase()) {
            case "ig":
                this.attributeSplitMeasure = new IGAttributeSplitMeasure();
                break;
            case "igr":
                IGAttributeSplitMeasure igAttributeSplitMeasure = new IGAttributeSplitMeasure();
                igAttributeSplitMeasure.setUseGain(false);
                this.attributeSplitMeasure = igAttributeSplitMeasure;
                break;
            case "chi":
                this.attributeSplitMeasure = new ChiSquaredAttributeSplitMeasure();
                break;
            case "gini":
                this.attributeSplitMeasure = new GiniAttributeSplitMeasure();
                break;
            default:
                throw new Exception();
        }
    }

    @Override
    @SuppressWarnings("unchecked")
    public void buildClassifier(Instances data) throws Exception {
        // Create each classifier in the ensemble
        ensemble = new CourseworkTree[numTrees];
        for (int i = 0; i < numTrees; i++) {
            ensemble[i] = new CourseworkTree();
            ensemble[i].setAttSplitMeasure(attributeSplitMeasure);
        }

        // Attribute selection (exclude the class attribute)
        int attSelCount = (int)(attProp * (data.numAttributes()-1)); // Exclude the class attribute
        selectedAttributes = new Attribute[numTrees][attSelCount];
        selectedAttributeIndexes = (ArrayList<Integer>[]) new ArrayList[numTrees];
        removeAttributeIndexes = (ArrayList<Integer>[]) new ArrayList[numTrees];
        Random random = new Random();
        random.setSeed(seed);
        int randInt = 0;
        // Assign the attributes for each classifier in the ensemble
        for (int treeIt = 0; treeIt < numTrees; treeIt++) {
            selectedAttributes[treeIt] = new Attribute[attSelCount];
            selectedAttributeIndexes[treeIt] = new ArrayList<>();
            for (int attIt = 0; attIt < attSelCount; attIt++) {
                // Must be an attribute not already selected (no replacement attribute selection)
                while (selectedAttributeIndexes[treeIt].contains(randInt)) {
                    randInt = random.nextInt(data.numAttributes()-1);
                }
                selectedAttributes[treeIt][attIt] = data.attribute(randInt);
                selectedAttributeIndexes[treeIt].add(randInt);
            }
            selectedAttributeIndexes[treeIt].add(data.classIndex()); // Cannot remove the class attribute

            removeAttributeIndexes[treeIt] = new ArrayList<>();
            // Generate new data with selected attributes
            Remove removeFilter = new Remove();
            String removeString = "";
            for (int attIt = 0; attIt < data.numAttributes(); attIt++) {
                if (!selectedAttributeIndexes[treeIt].contains(attIt)) {
                    removeString += (attIt+1) + ","; // Remove filter uses indexing from 1
                    removeAttributeIndexes[treeIt].add(attIt);
                }
            }
            if (!removeString.isEmpty()) {
                removeString = removeString.substring(0, removeString.length() - 1); // Remove extra comma "," character
            }
            String[] options = new String[]{"-R", removeString};
            removeFilter.setOptions(options);
            removeFilter.setInputFormat(data);
            Instances subData = Filter.useFilter(data, removeFilter);

            // Now build the classifier with the given data (subset of attributes)
            ensemble[treeIt].buildClassifier(subData);
        }
    }

    public double classifyInstance(Instance instance) {
        double[] distribution = distributionForInstance(instance);
        double max = distribution[0];
        double index = 0;
        for (int i = 1; i < distribution.length; i++) {
            if (max < distribution[i]) {
                max = distribution[i];
                index = i;
            }
        }
        return index;
    }

    /**
     * Gets the distribution for the given instance. Can use average distributions which averages all the base
     * classifiers distributions for the given instance. Otherwise, uses majority voting.
     * @param instance the instance to be classified
     * @return double array of probability distributions
     */
    public double[] distributionForInstance(Instance instance) {
        int classCount = instance.numClasses();

        if (averageDistributions) {
            double[] averageDistribution = new double[classCount];
            for (int treeIt = 0; treeIt < numTrees; treeIt++) {
                // Copy of the instance with the given attributes for the current tree
                DenseInstance modInstance = new DenseInstance(instance);
                // Apply an offset, as removing an attribute will shorten it
                int offset = 0;
                for (int removeIndex : removeAttributeIndexes[treeIt]) {
                    modInstance.deleteAttributeAt(removeIndex - offset);
                    offset++;
                }

                double[] treeDistribution = ensemble[treeIt].distributionForInstance(modInstance);
                for (int classIt = 0; classIt < classCount; classIt++) {
                    averageDistribution[classIt] += treeDistribution[classIt];
                }
            }
            for (int classIt = 0; classIt < classCount; classIt++) {
                averageDistribution[classIt] /= numTrees;
            }
//            System.out.println(Arrays.toString(averageDistribution));
            return averageDistribution;
        } else {
            int[] votes = new int[classCount];

            for (int treeIt = 0; treeIt < numTrees; treeIt++) {
                // Copy of the instance with the given attributes for the current tree
                DenseInstance modInstance = new DenseInstance(instance);
                // Apply an offset, as removing an attribute will shorten it
                int offset = 0;
                for (int removeIndex : removeAttributeIndexes[treeIt]) {
                    modInstance.deleteAttributeAt(removeIndex - offset);
                    offset++;
                }
                votes[(int) ensemble[treeIt].classifyInstance(modInstance)]++;
            }

            double[] distribution = new double[classCount];
            for (int classIt = 0; classIt < classCount; classIt++) {
                distribution[classIt] = (double) votes[classIt] / numTrees;
            }
            return distribution;
        }
    }

    public static void main(String[] args) {
        Instances optdigitsData = WekaTools.loadLocalClassificationData("optdigits.arff");
        Instances chinatownData = WekaTools.loadLocalClassificationData("Chinatown.arff");

        // Random split 0.8 : 0.2 (training : test) split
        assert optdigitsData != null;
        assert chinatownData != null;
        Instances[] optidigitsSplit = WekaTools.splitDataRandom(optdigitsData, 0.8);
        Instances[] chinatownSplit = WekaTools.splitDataRandom(chinatownData, 0.8);

        TreeEnsemble treeEnsemble = new TreeEnsemble();
        treeEnsemble.setAverageDistributions(false);
        try {
            // Digits Data
            treeEnsemble.buildClassifier(optidigitsSplit[0]);

            System.out.println("optdigits data:");
            System.out.println("Accuracy: " + WekaTools.accuracy(treeEnsemble, optidigitsSplit[1]));
            System.out.println("Probabilities for first five test cases:");
            for (int i = 0; i < 5; i++) {
                System.out.println(Arrays.toString(treeEnsemble.distributionForInstance(optidigitsSplit[1].instance(i))));
            }

            // Chinatown data
            treeEnsemble.buildClassifier(chinatownSplit[0]);

            System.out.println("Chinatown data:");
            System.out.println("Accuracy:" + WekaTools.accuracy(treeEnsemble, chinatownSplit[1]));
            System.out.println("Probabilities for first five test cases:");
            for (int i = 0; i < 5; i++) {
                System.out.println(Arrays.toString(treeEnsemble.distributionForInstance(chinatownSplit[1].instance(i))));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
