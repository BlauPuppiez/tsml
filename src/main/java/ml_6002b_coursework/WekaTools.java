package ml_6002b_coursework;

import fileIO.OutFile;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.FileReader;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;

/**
 * Tools for using Weka Objects to calculate, load various information.
 */
public class WekaTools {
    /**
     * Numeric to normal attributes, which two values (i.e. binary nominal).
     * @return Normalised data.
     */
    public static Instances convertNumericToNominalBinary(Instances instances) {
        Instances normalisedData = new Instances(instances);
        for (int i = 0; i < normalisedData.numAttributes() - 1; i++) {
            if (normalisedData.attribute(i).isNumeric()) {
                double total = 0;
                for (Instance instance : normalisedData) {
                    total += instance.value(i);
                }
                double mean = total / normalisedData.numInstances();
                for (Instance instance : normalisedData) {
                    instance.setValue(i, instance.value(i) <= mean ? 0 : 1);
                }
            }
        }
        try {
            NumericToNominal convert = new NumericToNominal();
            convert.setInputFormat(normalisedData); // Original Data to convert

            Instances newData = Filter.useFilter(normalisedData, convert);
            return newData;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
    /**
     * Uses the classifier and actual (test) data to calculate accuracy of the classifier on the (test) data
     * @param c classifier
     * @param test data (or any data to test the classifier on)
     * @return accuracy value
     */
    public static double accuracy(Classifier c, Instances test) {
        int correct = 0;
        for (Instance instance : test) {
            double predVal = 0;
            try {
                predVal = c.classifyInstance(instance);
            } catch (Exception e) {
                System.out.println("Classification failed on " + instance);
                e.printStackTrace();
            }
            if (predVal == instance.classValue()) {
                correct++;
            }
        }
        return (double)correct/test.numInstances();
    }

    /**
     * Generate Instances class from the path to the data.
     * Class attribute is set to the last attribute.
     * @param fullPath to data.
     * @return Instances containing loaded data.
     */
    public static Instances loadClassificationData(String fullPath) throws Exception {
        Instances instances;
        try {
            FileReader reader = new FileReader(fullPath);
            instances = new Instances(reader);
            instances.setClassIndex(instances.numAttributes()-1);
            return instances;
        } catch (Exception e) {
            throw e;
        }
    }

    /**
     * Generate Instances class from the local path to the data.
     * Looks in the test_data folder (located in the same directory as this java file).
     * Class attribute is set to the last attribute.
     * @param localPath to data.
     * @return Instances containing loaded data.
     */
    public static Instances loadLocalClassificationData(String localPath) {
        Instances instances;
        try {
            String dataPath = new java.io.File(".").getCanonicalPath();
            dataPath += "\\src\\main\\java\\ml_6002b_coursework\\test_data\\"; // Add path to test_data directory
            FileReader reader = new FileReader(dataPath + localPath);
            instances = new Instances(reader);
            instances.setClassIndex(instances.numAttributes()-1);
            return instances;
        } catch (Exception e) {
            System.out.println("Error while trying to load classification data! " + e.toString());
        }
        return null;
    }

    public static void printInstancesInformation(Instances data) {
        System.out.println("Instance Count: " + data.numInstances());
        System.out.println("Attribute Count: " + data.numAttributes());
        System.out.println("Attributes:");
        Enumeration e = data.enumerateAttributes();
        while (e.hasMoreElements()) {
            System.out.println(" " + e.nextElement());
        }
        System.out.println("Classes: " + data.numClasses());
        System.out.println("Class Attribute: " + data.classAttribute());
    }

    /**
     * Splits data into specified proportions
     * @param all the data to split.
     * @param proportion to keep in training, i.e. index 0;
     * @return Instances[] of length 2, index 0 for the specified train, and index 1 for the specified test data.
     */
    public static Instances[] splitData(Instances all, double proportion) {
        int trainCount = (int)Math.round(all.numInstances() * proportion);
        Instances[] split = new Instances[2];
        split[0] = new Instances(all, 0, trainCount); // Train
        split[1] = new Instances(all, trainCount, all.numInstances() - trainCount); // Test
        return split;
    }

    /**
     * Splits data into specified proportions, with randomisation.
     * @param all the data to split.
     * @param proportion to keep in training, i.e. index 0;
     * @return Instances[] of length 2, index 0 for the specified train, and index 1 for the specified test data.
     */
    public static Instances[] splitDataRandom(Instances all, double proportion) {
        Instances randomisedAll = new Instances(all, 0, all.numInstances()); // Copy
        randomisedAll.randomize(new Random()); // Randomise instance positions
        return splitData(randomisedAll, proportion);
    }

    public static Instances resampleNoReplacement(Instances data, double sampleFactor) throws Exception {
        Instances sampledInstances;
        if (sampleFactor == 1) {
            sampledInstances = new Instances(data);
        } else if (sampleFactor > 1) {
            throw new Exception("No replacement sampling cannot have a sample factor greater than 1!");
        } else if (sampleFactor <= 0) {
            throw new Exception("Sampling cannot have 0 or less samples!");
        } else {
            int instanceCount = (int)(data.numInstances() * sampleFactor);
            Instances remaining = new Instances(data);
            Random random = new Random();

            sampledInstances = new Instances(data, 0);
            while (sampledInstances.numInstances() < instanceCount) {
                int next = random.nextInt(remaining.numInstances());
                Instance randomInstance = remaining.get(next);
                sampledInstances.add(randomInstance);
                remaining.remove(next);
            }
        }
        return sampledInstances;
    }

    public static Instances resampleNoReplacement(Instances data, int sampleCount) throws Exception {
        Instances sampledInstances;
        if (sampleCount == data.numInstances()) {
            sampledInstances = new Instances(data);
        } else if (sampleCount > data.numInstances()) {
            throw new Exception("No replacement sampling cannot generate more samples than there are!");
        } else if (sampleCount <= 0) {
            throw new Exception("Must return at least 1 instance!");
        } else {
            sampledInstances = new Instances(data);
        }
        return sampledInstances;
    }

    public static double[] distribution(Instances data) {
        int classCount = data.numClasses();
        double[] distribution = new double[classCount];
        for (Instance instance : data) {
            distribution[(int)instance.classValue()]++;
        }
        int instanceCount = data.numInstances();
        for (int distIt = 0; distIt < classCount; distIt++) {
            distribution[distIt] /= instanceCount;
        }
        return distribution;
    }

    /**
     * Produces a confusion matrix from the predicted and actual values.
     * Predicted and actual must be of the same size!
     * @param predicted values.
     * @param actual values.
     * @return 2D int array confusion matrix.
     */
    public static int[][] confusionMatrix(int[] actual, int[] predicted, int numClasses) {
        // We set the 'x values' for actual.
        int[][] confusionMatrix = new int[numClasses][numClasses];
        int valueCount = actual.length;
        for (int valIt = 0; valIt < valueCount; valIt++) { // Value Iterator
            confusionMatrix[actual[valIt]][predicted[valIt]]++;
        }
        return confusionMatrix;
    }

    public static int[] classifyInstances(Classifier c, Instances test) {
        int[] predValues = new int[test.numInstances()];
        int instanceIt = 0;
        for (Instance instance : test) {
            try {
                double predVal = c.classifyInstance(instance);
                predValues[instanceIt] = (int)predVal;
                instanceIt++;
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return predValues;
    }

    public static int[] getClassValues(Instances data) {
        int[] actualValues = new int[data.numInstances()];
        int instanceIt = 0;
        for (Instance instance : data) {
            actualValues[instanceIt] = (int)instance.classValue();
            instanceIt++;
        }
        return actualValues;
    }

    public static void saveResults(Classifier c, Instances data, String fullPath) throws Exception {
        String probName = data.relationName();
        OutFile out = new OutFile(fullPath);
        out.writeLine(probName + "," + c.getClass().getSimpleName());
        out.writeLine("No parameter info");
        out.writeLine(String.valueOf(accuracy(c, data)));
        for (Instance instance : data) {
            StringBuilder nextString = new StringBuilder();
            int actual = (int)instance.classValue();
            nextString.append(actual);
            nextString.append(",");
            int prediction = (int)c.classifyInstance(instance);
            nextString.append(prediction);
            nextString.append(",");
            double[] distribution = c.distributionForInstance(instance);
            for (double prob : distribution) {
                nextString.append(",").append(prob);
            }
            out.writeLine(nextString.toString());
        }
        System.out.println("Saved results to file");
    }

    public static String getInstanceResult(Classifier c, Instance instance) throws Exception {
        StringBuilder result = new StringBuilder();
        result.append(instance.classValue()).append(",");
        result.append(c.classifyInstance(instance)).append(",");
        double[] distribution = c.distributionForInstance(instance);
        for (double prob : distribution) {
            result.append(",").append(prob);
        }
        return result.toString();
    }

    public static void saveResultsToFile(String fullPath, String name, String classifierName, String[] results) {
        OutFile out = new OutFile(fullPath);
        out.writeLine(name + "," + classifierName);
        out.writeLine("No parameter info");
        out.writeLine("Blank");
        for (String result : results) {
            out.writeLine(result);
        }
        System.out.println("Saved results to: " + fullPath);
    }

}
