package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;

public class GiniAttributeSplitMeasure extends AttributeSplitMeasure {

    /**
     * Computes the quality on splitting on the given attribute on the given data using the chi squared statistic.
     * @param data data to assess split attribute
     * @param att attribute to split on
     * @return Double value of the measurement of quality for the given data on splitting with the given attribute using
     * Information gain (ratio)
     * @throws Exception
     */
    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {
        int[][] contingencyTable = getContingencyTable(data, att);

        return AttributeMeasures.measureGini(contingencyTable);
    }

    /**
     * Main method.
     *
     * @param args the options for the split measure main
     */
    public static void main(String[] args) {
        try {
            // Load in the whisky data (full path for local path, stored in test_data folder)
//            FileReader reader = new FileReader("C:\\Users\\siuhu\\OneDrive\\Uni\\Year 3\\Machine Learning\\Coursework\\whisky.arff");
//            Instances instances = new Instances(reader);
//            instances.setClassIndex(instances.numAttributes() - 1);
            // Or using local path, comment as needed
            Instances instances = WekaTools.loadLocalClassificationData("whisky.arff");

            Attribute peatyAtt = instances.attribute(0);
            Attribute woodyAtt = instances.attribute(1);
            Attribute sweetAtt = instances.attribute(2);
            // Debug line for double-checking, should print in the same order as here
//            System.out.println(peatyAtt);
//            System.out.println(woodyAtt);
//            System.out.println(sweetAtt);

            GiniAttributeSplitMeasure igAttributeSplitMeasure = new GiniAttributeSplitMeasure();

            double peatyMeasure = igAttributeSplitMeasure.computeAttributeQuality(instances, peatyAtt);
            double woodyMeasure = igAttributeSplitMeasure.computeAttributeQuality(instances, woodyAtt);
            double sweetMeasure = igAttributeSplitMeasure.computeAttributeQuality(instances, sweetAtt);

            System.out.println("measure Gini for attribute Peaty splitting diagnosis = " + peatyMeasure);
            System.out.println("measure Gini for attribute Woody splitting diagnosis = " + woodyMeasure);
            System.out.println("measure Gini for attribute Sweet splitting diagnosis = " + sweetMeasure);
        } catch (Exception e) {
            System.out.println("Error!");
        }
    }
}
