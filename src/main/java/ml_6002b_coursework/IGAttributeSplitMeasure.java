package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;


public class IGAttributeSplitMeasure extends AttributeSplitMeasure {

    /** Use Information gain (true) or Information gain ratio (false) */
    private boolean useGain = true;

    /**
     * Computes the quality on splitting on the given attribute on the given data using information gain or information
     * gain ratio.
     * @param data data to assess split attribute.
     * @param att attribute to split on.
     * @return Double value of the measurement of quality for the given data on splitting with the given attribute using Information gain
     * (ratio).
     * @throws Exception
     */
    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {
        int classCount = data.numClasses();
        int attValues = att.numValues();

        int[][] contingencyTable = new int[attValues][classCount];

        for (Instance instance : data) {
            int attValue = (int)instance.value(att);
            int classValue = (int)instance.classValue();
            contingencyTable[attValue][classValue]++;
        }

        return useGain ? AttributeMeasures.measureInformationGain(contingencyTable) : AttributeMeasures.measureInformationGainRatio(contingencyTable);
    }

    /**
     * Select between information gain (true) and information gain ratio (false).
     */
    public void setUseGain(boolean useGain) {
        this.useGain = useGain;
    }

    /**
     * Main method.
     *
     * @param args the options for the split measure main
     */
    public static void main(String[] args) {
        // Load in the whisky data
        Instances instances = null;
        try {
            FileReader reader = new FileReader("C:\\Users\\siuhu\\OneDrive\\Uni\\Year 3\\Machine Learning\\Coursework\\whisky.arff");
            instances = new Instances(reader);
            instances.setClassIndex(instances.numAttributes()-1);
        } catch (Exception e) {
            System.out.println("Error!");
        }

        assert instances != null;

        Attribute peatyAtt = instances.attribute(0);
        Attribute woodyAtt = instances.attribute(1);
        Attribute sweetAtt = instances.attribute(2);

        // Debug lines; should print in the same order as here
//            System.out.println(peatyAtt);
//            System.out.println(woodyAtt);
//            System.out.println(sweetAtt);


        IGAttributeSplitMeasure igAttributeSplitMeasure = new IGAttributeSplitMeasure();

        try {
            double peatyMeasure = igAttributeSplitMeasure.computeAttributeQuality(instances, peatyAtt);
            double woodyMeasure = igAttributeSplitMeasure.computeAttributeQuality(instances, woodyAtt);
            double sweetMeasure = igAttributeSplitMeasure.computeAttributeQuality(instances, sweetAtt);

            System.out.println("measure Information Gain for attribute Peaty splitting diagnosis = " + peatyMeasure);
            System.out.println("measure Information Gain for attribute Woody splitting diagnosis = " + woodyMeasure);
            System.out.println("measure Information Gain for attribute Sweet splitting diagnosis = " + sweetMeasure);

            // And also using information gain ratio
            igAttributeSplitMeasure.setUseGain(false);
            peatyMeasure = igAttributeSplitMeasure.computeAttributeQuality(instances, peatyAtt);
            woodyMeasure = igAttributeSplitMeasure.computeAttributeQuality(instances, woodyAtt);
            sweetMeasure = igAttributeSplitMeasure.computeAttributeQuality(instances, sweetAtt);

            System.out.println("measure Information Gain Ratio for attribute Peaty splitting diagnosis = " + peatyMeasure);
            System.out.println("measure Information Gain Ratio for attribute Woody splitting diagnosis = " + woodyMeasure);
            System.out.println("measure Information Gain Ratio for attribute Sweet splitting diagnosis = " + sweetMeasure);
        } catch (Exception ignored) {

        }
    }
}
