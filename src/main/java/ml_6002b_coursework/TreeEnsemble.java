package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Copy;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class TreeEnsemble extends AbstractClassifier {
    private CourseworkTree[] ensemble;
    /** Selected attributes for each classifier in the ensemble */
    private Attribute[][] selectedAttributes;
    private ArrayList<Integer>[] selectedAttributeIndexes;

    /** Number of trees in the ensemble */
    private int numTrees = 50;

    /** Proportion of attributes to use in each classifier, does so without replacement */
    private double attProp = 0.5;

    public void setNumTrees(int numTrees) {
        if (numTrees < 1) {

        } else {
            this.numTrees = numTrees;
        }
    }

    public void setAttProp(double numAttSelec) {
        if (attProp <= 0) {

        } else if (attProp > 1) {

        } else {
            attProp = numAttSelec;
        }
    }

    @Override
    @SuppressWarnings("unchecked")
    public void buildClassifier(Instances data) throws Exception {
        ensemble = new CourseworkTree[numTrees];
        for (int i = 0; i < numTrees; i++) {
            ensemble[i] = new CourseworkTree();
        }

        int attSelecCount = (int)(attProp * (data.numAttributes()-1));// Exclude the class attribute
        selectedAttributes = new Attribute[numTrees][attSelecCount];
        selectedAttributeIndexes = (ArrayList<Integer>[]) new ArrayList[numTrees];
        Random random = new Random();
        int randInt = 0;
        // Assign the attributes for each classifier in the ensemble
        for (int treeIt = 0; treeIt < numTrees; treeIt++) {
            selectedAttributes[treeIt] = new Attribute[attSelecCount];
            selectedAttributeIndexes[treeIt] = new ArrayList<>();
            for (int attIt = 0; attIt < attSelecCount; attIt++) {
                // Must be an attribute not already selected (no replacement attribute selection)
                while (selectedAttributeIndexes[treeIt].contains(randInt)) {
                    randInt = random.nextInt(attSelecCount);
                }
                selectedAttributes[treeIt][attIt] = data.attribute(randInt);
                selectedAttributeIndexes[treeIt].add(randInt);
            }
            // Now add the class attribute?
            // And now build the tree with the selected attributes
            Copy convert = new Copy();
            AttributeSelection attributeSelector = new AttributeSelection();
            // Set options to copy only certain attributes
            String[] options = new String[2];
            options[0] = "-P";
            StringBuilder indexString = new StringBuilder();
            for (Integer attIndex : selectedAttributeIndexes[treeIt]) {
                int index = attIndex + 1;
                indexString.append(index).append(",");
            }
            options[1] = indexString.substring(0, indexString.length()-1); // Remove extra ',' character
            System.out.println("OPTOINS: " + options[1]);
            convert.setOptions(options);
            attributeSelector.setOptions(options);
            convert.setInputFormat(data); // Original Data to convert
            attributeSelector.setInputFormat(data);

            Instances newData = Filter.useFilter(data, attributeSelector);
            System.out.println(newData.numAttributes());

            //ensemble[treeIt].buildClassifier(data);
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

    public double[] distributionForInstance(Instance instance) {
        return new double[]{0.0, 0.0};
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
        try {
            treeEnsemble.buildClassifier(chinatownSplit[0]);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
